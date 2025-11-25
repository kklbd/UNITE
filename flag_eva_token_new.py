from typing import cast, List, Union, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from modeling_evaluation_base import BGE_AudioToken
import os
from PIL import Image
from torch.utils.data import DataLoader
from data_ds_cirr import Multimodal_Dataset,Multimodal_Collator
from flag_mm_dataset import MMIT_Dataset, MMIT_Collator, Image_Dataset, Image_Collator,Audio_Dataset,Audio_Collator,text_Dataset,text_Collator
class Flag_bgev_model:
    def __init__(
            self,
            model_name_bge: str = "BAAI/bge-m3",
            model_name_eva: str = "EVA02-CLIP-B-16",
            normlized: bool = True,
            eva_pretrained_path: str = "eva_clip",
            resume_path = None,
            pooling_method: str = 'cls',
            use_fp16: bool=True,
            image_dir: str = None,
            audio_dir: str = None,
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_bge)

        self.model = BGE_AudioToken(model_name_bge = model_name_bge,
                        # model_name_eva = model_name_eva, # "EVA02-CLIP-B-16",
                        normlized = True,
                        eva_pretrained_path = eva_pretrained_path,)

        self.model.load_state_dict(torch.load(resume_path, map_location='cpu'))

        self.normalize_embeddings = normlized
        self.pooling_method = pooling_method

        self.image_dir = image_dir
        self.audio_dir = audio_dir

        if use_fp16: 
            self.use_fp16 = True
            self.model.half()
        else:
            self.use_fp16 = False
            
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)


    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int=256,
                       max_length: int=512,
                       query_type: str = None,
                       ) -> np.ndarray:
        
        if query_type == 'text':        

            input_texts = queries
            
            return self.encode_text(input_texts, batch_size=batch_size, max_length=max_length)
        elif query_type == 'mm_it':
            q_text, q_img = queries
            
            input_texts = q_text
            
            return self.encode_mm_it(input_texts, q_img, batch_size=batch_size)
        elif query_type == 'image':
            return self.encode_image(q_img, batch_size=batch_size)
        
        elif query_type == 'audio':
            q_audio = queries
            return self.encode_audio(q_audio, batch_size=batch_size)
        else:
            raise NotImplementedError


    def encode_corpus(self,
                      corpus: dict,
                      batch_size: int=256,
                      max_length: int=512,
                      corpus_type: str = None,
                      ) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        if corpus_type == 'text':
            return self.encode_text(corpus["text"], batch_size=batch_size, max_length=max_length)
        elif corpus_type == 'mm_it':
            return self.encode_mm_it(corpus["text"], corpus["image"], batch_size=batch_size, max_length=max_length)
        elif corpus_type == 'image':
            return self.encode_image(corpus["image"], batch_size=batch_size, max_length=max_length)
        elif corpus_type == 'audio':
            return self.encode_audio(corpus["audio"], batch_size=batch_size, max_length=max_length)
        else:
            raise RuntimeError(f"You must choose a corpus type from: [mm_it, text, image]")
        


    # @torch.no_grad()
    # def encode_text(self, sentences: Union[List[str], str], batch_size: int=256, max_length: int=128) -> np.ndarray:
    #     if self.num_gpus > 0:
    #         batch_size = batch_size * self.num_gpus
    #     self.model.eval()

    #     input_was_string = False
    #     if isinstance(sentences, str):
    #         sentences = [sentences]
    #         input_was_string = True

    #     all_embeddings = []
    #     for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings", disable=len(sentences)<256):
    #         sentences_batch = sentences[start_index:start_index + batch_size]
    #         inputs = self.tokenizer(
    #             sentences_batch,
    #             padding=True,
    #             truncation=True,
    #             return_tensors='pt',
    #             max_length=max_length,
    #         ).to(self.device)
            
    #         embeddings = self.model.t_encoder(inputs) 
    #         embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    #         embeddings = cast(torch.Tensor, embeddings)
    #         all_embeddings.append(embeddings.cpu().numpy())

    #     all_embeddings = np.concatenate(all_embeddings, axis=0)
    #     if input_was_string:
    #         return all_embeddings[0]
    #     return all_embeddings
    #     @torch.no_grad()
    
    @torch.no_grad()
    def encode_text(self, sentences: Union[List[str], str], batch_size: int=256, max_length: int=128) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        text_dataset = text_Dataset( 
                                     sentences=sentences
                                     )
        text_collator = text_Collator(self.tokenizer, caption_max_len=77,device=self.device)

        text_dataloader = DataLoader(dataset=text_dataset, 
                                      collate_fn=text_collator, 
                                      num_workers=8, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,)




        for start_index in tqdm(text_dataloader, desc="Inference Text Embeddings"):
            
            
            embeddings = self.model.t_encoder(start_index) 
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


    @torch.no_grad()
    def encode_mm_it(self, captions: Union[List[str], str], image_ids: Union[List[str], str],  batch_size: int=256, max_length: int=512) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(captions, str):
            captions = [captions]
            image_ids = [image_ids]
            input_was_string = True

        all_embeddings = []
        mm_it_dataset = MMIT_Dataset(captions=captions, 
                                     image_ids=image_ids, 
                                     image_dir=self.image_dir,
                                     image_processor=self.model.preprocess_val
                                     )
        mm_it_collator = MMIT_Collator(self.tokenizer, caption_max_len=312)

        mm_it_dataloader = DataLoader(dataset=mm_it_dataset, 
                                      collate_fn=mm_it_collator, 
                                      num_workers=8, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,)

        for data in tqdm(mm_it_dataloader, desc="Inference Embeddings", disable=len(captions)<256):
            captions_inputs = data[0].to(self.device)
            
            
            images = data[1].to(self.device)
            if self.use_fp16 and images.dtype != torch.float16:
                images = images.half()

            embeddings = self.model.mm_encoder(images, captions_inputs)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings
    
    @torch.no_grad()
    def encode_image(self, image_ids: Union[List[str], str],  batch_size: int=256, max_length: int=512) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        all_embeddings = []
        image_dataset = Image_Dataset(image_ids=image_ids, 
                                     image_dir=self.image_dir,
                                     image_processor=self.model.preprocess_val
                                     )
        image_collator = Image_Collator(self.tokenizer, caption_max_len=312)

        image_dataloader = DataLoader(dataset=image_dataset, 
                                      collate_fn=image_collator, 
                                      num_workers=8, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,)

        for data in tqdm(image_dataloader, desc="Inference Image Embeddings"):
            
            images = data.to(self.device)
            if self.use_fp16 and images.dtype != torch.float16:
                images = images.half()
            


            embeddings = self.model.encode_image_only(images=images)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        return all_embeddings
    
    @torch.no_grad()
    def encode_audio(self, audio_ids: Union[List[str], str],  batch_size: int=256, max_length: int=512) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        all_embeddings = []
        audio_dataset = Audio_Dataset(audio_ids=audio_ids, 
                                     audio_dir=self.audio_dir,
                                     )
        audio_collator = Audio_Collator(self.model.a_model.modality_config['audio'],self.tokenizer)

        audio_dataloader = DataLoader(dataset=audio_dataset, 
                                      collate_fn=audio_collator, 
                                      num_workers=8, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,)

        for data in tqdm(audio_dataloader, desc="Inference Audio Embeddings"):
            
            audios = data.to(self.device)
            if self.use_fp16 and audios.dtype != torch.float16:
                audios = audios.half()


            embeddings = self.model.a_encoder(audios=audios)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        return all_embeddings

    @torch.no_grad()
    def encode_mm_it_mbeir(self, captions: Union[List[str], str], image_ids: Union[List[str], str],  batch_size: int=256, max_length: int=512) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(captions, str):
            captions = [captions]
            image_ids = [image_ids]
            input_was_string = True

        all_embeddings = []
        mm_it_dataset = MMIT_Dataset(captions=captions, 
                                     image_ids=image_ids, 
                                     image_dir=self.image_dir,
                                     image_processor=self.model.preprocess_val
                                     )
        mm_it_collator = MMIT_Collator(self.tokenizer, caption_max_len=128)

        mm_it_dataloader = DataLoader(dataset=mm_it_dataset, 
                                      collate_fn=mm_it_collator, 
                                      num_workers=8, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,)

        for data in tqdm(mm_it_dataloader, desc="Inference Embeddings", disable=len(captions)<256):
            captions_inputs = data[0].to(self.device)
            
            
            images = data[1].to(self.device)
            if self.use_fp16 and images.dtype != torch.float16:
                images = images.half()


            text_embeddings = self.model.encode_text(captions_inputs)
            image_embeddings = self.model.encode_image_only(images)
          
            
            embeddings = text_embeddings + image_embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

class FlagReranker:
    def __init__(
            self,
            model_name_or_path: str = None,
            use_fp16: bool = False
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

        if use_fp16: self.model.half()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        self.model.eval()

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    @torch.no_grad()
    def compute_score(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], batch_size: int = 256,
                      max_length: int = 512) -> List[float]:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        for start_index in tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores",
                                disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores


