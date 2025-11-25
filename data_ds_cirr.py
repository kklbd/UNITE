import os.path
import random
from dataclasses import dataclass
import numpy as np
import datasets
from torch.utils.data import Dataset, IterableDataset
import sys
sys.path.append("/home/ls/LanguageBind/languagebind/audio")
from processing_audio import make_list_of_images,load_and_transform_audio,get_audio_transform,AudioTransform,torchaudio_loader
from arguments import DataArguments
from torchvision import transforms
from PIL import Image
import json
import torch
import torch.distributed
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler,RandomClipSampler
import logging
import csv
import librosa
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from transformers import ProcessorMixin
import torch
import torchaudio
import random
import numpy as np
torchaudio.set_audio_backend("soundfile")


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
    # 如果 waveform 太短，会触发 torchaudio 的断言，所以我们手动补零
    if waveform.shape[1] < 400:
        pad_len = 400 - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

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
    # if abs(p) / n_frames > 0.2:
    #     logging.warning(
    #         "Large gap between audio n_frames(%d) and "
    #         "target_length (%d). Is the audio_target_length "
    #         "setting correct?",
    #         n_frames,
    #         target_length,
    #     )
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
            # waveform_clip=augment_waveform(waveform_clip, sample_rate)
            waveform_melspec = waveform2melspec(                              # waveform_melspec: torch.Size([1, 128, 204])
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)                  

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]   

        all_clips = torch.stack(all_clips, dim=0)           #(clips_per_video, L , num_mel_bins, target_length)
        audio_outputs.append(all_clips)                 

    return torch.stack(audio_outputs, dim=0)        #(batch_size, clips_per_video, L, num_mel_bins, target_length)

# def load_and_transform_audio_data(
#     audio_paths,
#     device,
#     num_mel_bins=128,
#     target_length=204,       # 你原模型要求的帧数
#     sample_rate=16000,
#     mean=-4.268,
#     std=9.138,
# ):
#     if audio_paths is None:
#         return None

#     audio_outputs = []

#     for audio_path in audio_paths:
#         waveform, sr = torchaudio.load(audio_path)

#         if sample_rate != sr:
#             waveform = torchaudio.functional.resample(
#                 waveform, orig_freq=sr, new_freq=sample_rate
#             )

#         waveform = waveform.float()
#         waveform = waveform - waveform.mean()

#         # 提取整段 fbank（不管时长多少，先提完）
#         fbank = torchaudio.compliance.kaldi.fbank(
#             waveform,
#             htk_compat=True,
#             sample_frequency=sample_rate,
#             use_energy=False,
#             window_type="hanning",
#             num_mel_bins=num_mel_bins,
#             dither=0.0,
#             frame_length=25,
#             frame_shift=10,
#         )  # shape: [T, mel_bins]

#         fbank = fbank.transpose(0, 1)  # [mel_bins, T]
#         fbank = fbank.unsqueeze(0).unsqueeze(0)  # [1, 1, mel_bins, T]

#         # 插值下采样到目标长度（time=204）
#         fbank = F.interpolate(fbank, size=(num_mel_bins, target_length), mode="bilinear", align_corners=False)
#         fbank = fbank.squeeze(0)  # [1, mel_bins, target_length]

#         normalize = transforms.Normalize(mean=mean, std=std)
#         fbank = normalize(fbank).to(device)

#         audio_outputs.append(fbank)

#     audio_outputs = torch.stack(audio_outputs, dim=0)  # [B, 1, 128, 204]
#     audio_outputs = audio_outputs.unsqueeze(1)         # [B, 1, 1, 128, 204]
#     return audio_outputs

# def load_and_transform_audio_data(
#     audio_paths,
#     device,
#     num_mel_bins=128,
#     target_length=204,        # 每段片段提出来还是 204 帧
#     sample_rate=16000,
#     clip_duration=2.0,        # 每段采 2 秒
#     num_clips=5,              # 每条音频采多少段
#     mean=-4.268,
#     std=9.138,
# ):
#     if audio_paths is None:
#         return None

#     audio_outputs = []

#     for audio_path in audio_paths:
#         waveform, sr = torchaudio.load(audio_path)
#         if sr != sample_rate:
#             waveform = torchaudio.functional.resample(
#                 waveform, orig_freq=sr, new_freq=sample_rate
#             )

#         waveform = waveform.float()
#         total_samples = waveform.shape[1]
#         clip_samples = int(clip_duration * sample_rate)

#         clips = []

#         for _ in range(num_clips):
#             if total_samples <= clip_samples:
#                 start = 0
#                 padded = torch.zeros((1, clip_samples))
#                 padded[:, :total_samples] = waveform
#                 clip = padded
#             else:
#                 start = random.randint(0, total_samples - clip_samples)
#                 clip = waveform[:, start:start + clip_samples]

#             # 提取 fbank 特征
#             fbank = torchaudio.compliance.kaldi.fbank(
#                 clip,
#                 htk_compat=True,
#                 sample_frequency=sample_rate,
#                 use_energy=False,
#                 window_type="hanning",
#                 num_mel_bins=num_mel_bins,
#                 dither=0.0,
#                 frame_length=25,
#                 frame_shift=10,
#             )  # shape: [T, mel_bins]

#             fbank = fbank.transpose(0, 1)  # shape: [mel_bins, T]

#             # pad or truncate 到 target_length=204
#             cur_len = fbank.shape[1]
#             if cur_len < target_length:
#                 pad_len = target_length - cur_len
#                 fbank = F.pad(fbank, (0, pad_len), mode="constant", value=0)
#             elif cur_len > target_length:
#                 fbank = fbank[:, :target_length]

#             fbank = fbank.unsqueeze(0)  # [1, mel_bins, target_len]
#             clips.append(fbank)

#         clips = torch.stack(clips, dim=0)  # [num_clips, 1, mel_bins, target_len]
#         pooled = clips.mean(dim=0)        # [1, mel_bins, target_len]

#         normalize = transforms.Normalize(mean=mean, std=std)
#         pooled = normalize(pooled).to(device)

#         audio_outputs.append(pooled)

#     audio_outputs = torch.stack(audio_outputs, dim=0)  # [B, 1, 128, 204]
#     audio_outputs = audio_outputs.unsqueeze(1)         # [B, 1, 1, 128, 204]
#     return audio_outputs





class Multimodal_Dataset(Dataset):
    def __init__(
            self,
            args: DataArguments,
            image_processor=None
    ):
        if os.path.isdir(args.train_data):
            annotation_path = os.path.join(args.train_data, "dataset")
            self.audio_dir = os.path.join(args.train_data, "train")


            self.audio_file_list = [] #只放名称，不放完整路径
            self.caption_list = [] # [[,], [,]] 多个caption
            
            with open(os.path.join(annotation_path, "clotho_captions_development新.csv"), newline='', encoding='utf-8') as fp:
                train_dict = csv.reader(fp)
                next(train_dict)# 跳过第一行（表头）
                for row in train_dict:
                    audio_file = row[0]
                    
                    captions = [caption for caption in row[1:6] if caption]  # 去除空字符串

                    self.audio_file_list.append(audio_file)
                    self.caption_list.append(captions)
        else:
            raise RuntimeError("Error: Please set arg.train_data to dataset directory.")

        
        self.args = args
        self.total_len = len(self.audio_file_list)

    def __len__(self):
        return self.total_len


    def __getitem__(self, item): # -> Tuple[BatchEncoding, List[BatchEncoding]]:
        audio_path = os.path.join(self.audio_dir, self.audio_file_list[item])
        
        
        caption = random.choice(self.caption_list[item]) # five captions per image, and only choose one
                
        return caption, audio_path
    

class Multimodal_Collator(ProcessorMixin):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.

    About caption_max_len, the BGE-Base can handle 512 tokens, but still set 77 here.
    Due to the long token squence is important for cuda memory, shorter length can use bigger batch size. 
    But we can use longer caption when training on the specific long caption datasets.
    """
    attributes = []
    tokenizer_class = ("LanguageBindAudioTokenizer")
    def __init__(self,config,tokenizer=None,**kwargs):
        super().__init__(**kwargs)
        
        self.config = config
        self.transform = get_audio_transform(config)
        self.audio_processor = load_and_transform_audio
        self.tokenizer = tokenizer
    def __call__(self, features,caption_max_len=77, return_tensors=None, **kwargs):
        caption = [f[0] for f in features]
        audio_paths = [f[1] for f in features]
        
        c_collated = self.tokenizer(
            caption,
            padding=True,
            truncation=True,
            max_length=caption_max_len,
            return_tensors="pt",
        )
        device = torch.device("cpu")
        # print(f"Key444444, ",audio_paths)
        # print(f"Key4444445, ",type(audio_paths))
        # a_collated=load_and_transform_audio_data(audio_paths, device) #(3, 3, 1, 128, 204)即(batch_size, clips_per_video, L, num_mel_bins, target_length)
        audios = make_list_of_images(audio_paths)
        audio_features = [self.audio_processor(audio, self.transform) for audio in audios]
        audio_features = torch.stack(audio_features)
        # print(f"Key3333333, Type of value: {type(audio_features)}")
        a_collated = audio_features
        # print(f"a_collated.shape: {a_collated.shape}")
        # print(f"Key2222222, Type of value: {type(a_collated)}")
        return {"captions": c_collated, "audios": a_collated}


# class Multimodal_Dataset(Dataset):
#     def __init__(self, args:DataArguments, image_processor=None) -> None:
#         self.image_dir = os.path.join(args.train_data_image, "CIRR_images")
#         self.train_group_size = args.train_group_size
        
#         jsonl_dir = args.train_data 
#         cirr_data_path = os.path.join(jsonl_dir, "cirr/query_train.jsonl")
#         self.hn_mining = False # True if use "cirr/query_train_hn_mining.jsonl"
        
#         self.cirr_dataset = datasets.load_dataset('json', data_files=cirr_data_path, split='train')    
        
#         self.total_len = len(self.cirr_dataset)
          
#         self.image_processor = image_processor
        
#     def img2pil(self, image_path):
#         complelte_img_path = os.path.join(self.image_dir, image_path)
#         return Image.open(complelte_img_path)
    
#     def __getitem__(self, item):
#         q_img = self.cirr_dataset[item]["q_img"]
#         q_text = self.cirr_dataset[item]["q_text"]
#         q_img = self.image_processor(self.img2pil(q_img))
        
#         positive_img = self.cirr_dataset[item]["positive_value"]
#         positive_img = [self.image_processor(self.img2pil(positive_img))]
#         if not self.hn_mining:
#             hn_images = random.sample(self.cirr_dataset[item]["hn_image"], self.train_group_size - 1)
#         else:
#             per_select_num = (self.train_group_size - 1) // 2
#             hn_images_1 = random.sample(self.cirr_dataset[item]["hn_image"], per_select_num)
#             hn_images_2 = random.sample(self.cirr_dataset[item]["hn_mining_images"][:20], per_select_num)
            
#             hn_images = hn_images_1 + hn_images_2
#         hn_images = [self.image_processor(self.img2pil(_hn)) for _hn in hn_images]
        
        
        
#         image_candidates = positive_img + hn_images    
#         return q_img, q_text, image_candidates
        
    
#     def __len__(self):
#         return self.total_len


# class Multimodal_Collator:
#     def __init__(self, tokenizer, mmit_max_len=109, pure_text_max_len=256):
#         self.tokenizer = tokenizer
#         self.mmit_max_len = mmit_max_len
#         self.text_max_len = pure_text_max_len
    
#     def reshape_image_candidate(self, i_candidates):
#         all_candidates = []
#         for group in i_candidates:
#             for image in group:
#                 all_candidates.append(image)
#         return all_candidates
    
#     def reshape_text_candidate(self, t_candidates):
#         all_candidates = []
#         for group in t_candidates:
#             for text in group:
#                 all_candidates.append(text)
#         return all_candidates
    
#     def reshape_mmit_candidate(self, mm_candidates):
#         all_candidates = []
#         for group in mm_candidates:
#             for mm in group:
#                 all_candidates.append(mm)
#         return all_candidates
    
    
#     def __call__(self, features):
        
#         q_images = [f[0] for f in features]
#         q_texts = [f[1] for f in features]
#         image_candidates = [f[2] for f in features]
        
        
        
#         q_text_collated = self.tokenizer(
#             q_texts,
#             padding= True, #"max_length",
#             truncation=True,
#             max_length=self.mmit_max_len,
#             return_tensors="pt",
#         )
#         q_image_collated = torch.stack(q_images)
        
        
#         c_images = self.reshape_image_candidate(image_candidates)
#         c_image_collated = torch.stack(c_images)

#         return {"mm_it_query": (q_image_collated, q_text_collated), "image_candidate": c_image_collated}
    