import math
import os.path
import random
from dataclasses import dataclass
from typing import Iterator
from torchvision import transforms
import datasets
from torch.utils.data import Dataset, IterableDataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import CLIPImageProcessor
# from arguments import DataArguments
import sys
sys.path.append("/home/ls/LanguageBind/languagebind/audio")
from processing_audio import make_list_of_images,load_and_transform_audio,get_audio_transform,AudioTransform,torchaudio_loader
sys.path.append("/home/ls/FlagEmbedding/research/visual_bge/visual_bge")
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler,RandomClipSampler
from PIL import Image
import json
import torch
import torch.distributed
import logging
from io import BytesIO
import warnings
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import csv
import librosa
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from transformers import ProcessorMixin
import random
import numpy as np
torchaudio.set_audio_backend("soundfile")
class MMIT_Dataset(Dataset):
    def __init__(self, captions, image_ids, image_dir, image_processor) -> None:
        img_id_example = image_ids[0]
        img_id_example = str(img_id_example)
        if img_id_example[-4:] in [".jpg", ".png"]:
            self.image_path =[os.path.join(image_dir, str(id)) for id in image_ids]
        else:
            warnings.warn("Not found file extention in image_ids, will forcefully add '.jpg'.", UserWarning)
            self.image_path =[os.path.join(image_dir, str(id) + '.jpg') for id in image_ids]
        self.captions = captions
        self.image_processor = image_processor
    
    def __getitem__(self, item):
        pil_data = Image.open(self.image_path[item])
        
        pil_data = pil_data.convert('RGB') 
        image = self.image_processor(pil_data)

        
        caption = self.captions[item]

        return caption, image

    def __len__(self):
        return len(self.image_path)


class MMIT_Collator:
    def __init__(self, tokenizer, caption_max_len):
        self.tokenizer = tokenizer
        self.caption_max_len = caption_max_len
    


    def __call__(self, features):
        caption = [f[0] for f in features]
        images = [f[1] for f in features]
        
        c_collated = self.tokenizer(
            caption,
            truncation=True,
            padding = True,
            max_length=self.caption_max_len,
            return_tensors="pt",
        )

        i_collated = torch.stack(images)    
       

        return c_collated, i_collated
    
class Image_Dataset(Dataset):
    def __init__(self, image_ids, image_dir, image_processor) -> None:

        self.image_path =[os.path.join(image_dir, str(id)) for id in image_ids]
        self.image_processor = image_processor
    
    def __getitem__(self, item):
        pil_data = Image.open(self.image_path[item])
        image = self.image_processor(pil_data)

        return image

    def __len__(self):
        return len(self.image_path)

class Image_Collator:
    def __init__(self, tokenizer, caption_max_len):
        self.tokenizer = tokenizer
        self.caption_max_len = caption_max_len
    

    def __call__(self, features):
        images = features
        i_collated = torch.stack(images)    
        return i_collated

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds
def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints

def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank

def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = RandomClipSampler(
        clip_duration=clip_duration
    )
    # clip_sampler = ConstantClipsPerVideoSampler(
    #     clip_duration=clip_duration, clips_per_video=clips_per_video
    # )

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)                          #(C, L)C为通道数，L为采样点数
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[                                   
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(                              # waveform_melspec: torch.Size([1, 128, 204])
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)                  

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]   

        all_clips = torch.stack(all_clips, dim=0)           #(clips_per_video, L , num_mel_bins, target_length)
        audio_outputs.append(all_clips)                 

    return torch.stack(audio_outputs, dim=0)        #(batch_size, clips_per_video, L, num_mel_bins, target_length)


class text_Dataset(Dataset):
    def __init__(
            self,
            sentences
    ):
        self.sentences=sentences
           
        self.total_len = len(self.sentences)

    def __len__(self):
        return self.total_len


    def __getitem__(self, item): 
        
        
        caption = self.sentences[item] 
                
        return caption

class text_Collator:
    def __init__(self, tokenizer, caption_max_len,device):
        self.tokenizer = tokenizer
        self.caption_max_len = caption_max_len
        self.device = device

    def __call__(self, features):
        caption = [f for f in features]
        
        c_collated = self.tokenizer(
            caption,
            padding=True,
            truncation=True,
            max_length=self.caption_max_len,
            return_tensors="pt",
        ).to(self.device)

        return c_collated

class Audio_Dataset(Dataset):
    def __init__(self, audio_ids, audio_dir) -> None:

        self.audio_path =[os.path.join(audio_dir, str(id)) for id in audio_ids]

    
    def __getitem__(self, item):

        audio_path = self.audio_path[item]
      
        return audio_path

    def __len__(self):
        return len(self.audio_path)

class Audio_Collator(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindAudioTokenizer")
    def __init__(self,config,tokenizer=None,**kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_audio_transform(config)
        self.audio_processor = load_and_transform_audio
        self.tokenizer = tokenizer

    def __call__(self, features,caption_max_len=77, return_tensors=None, **kwargs):
        audio_paths = [f for f in features]
        
        device = torch.device("cpu")
        audios = make_list_of_images(audio_paths)
        audio_features = [self.audio_processor(audio, self.transform) for audio in audios]
        audio_features = torch.stack(audio_features)
        # print(f"Key3333333, Type of value: {type(audio_features)}")
        a_collated = audio_features

        return a_collated