import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5,6,7,8"
os.environ["WANDB_MODE"] = "disabled"
import logging
import os
from pathlib import Path
from transformers.tokenization_utils_base import BatchEncoding

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
import sys
print(os.getcwd())
sys.path.append('/home/ls/FlagEmbedding/research/baai_general_embedding/retromae_pretrain')
from arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from data_ds_cirr import Multimodal_Dataset, Multimodal_Collator
from modeling_ds_cirr import BGE_AudioToken
from trainer import PreTrainer
import torch
from torch.utils.tensorboard import SummaryWriter
from best_checkpoint_callback import SaveBestCheckpointCallback

logger = logging.getLogger(__name__)

from dataclasses import dataclass


def main():
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)



    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.bge_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.bge_model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)
    
    writer = SummaryWriter(os.path.join(training_args.output_dir, "my_tfwriter"))
    model = BGE_AudioToken(model_name_bge = model_args.bge_model_name_or_path,
                                # model_name_eva = model_args.visual_model_name_or_path, # "EVA02-CLIP-B-16",
                                normlized = training_args.normlized,
                                sentence_pooling_method = training_args.sentence_pooling_method,
                                negatives_cross_device = training_args.negatives_cross_device,
                                temperature = training_args.temperature,
                                writer=writer,
                                eva_pretrained_path = "eva_clip",)
    model = model.to(torch.float16)  # 强制转换模型为 float16
    fp16_count = sum(p.dtype == torch.float16 for p in model.parameters())
    fp32_count = sum(p.dtype == torch.float32 for p in model.parameters())

    with open('all_model_param_names.txt', 'w') as f:
        for name, param in model.named_parameters():
            f.write(f'{name} | shape: {tuple(param.shape)}\n')


    # print(f"FP16 parameters: {fp16_count}, FP32 parameters: {fp32_count}")
    # 打开一个文件用于保存参数的数据类型
    with open("model_parameter_dtypes.txt", "w") as file:
        for param in model.parameters():
            # 将每个参数的数据类型写入文件
            file.write(f"{param.dtype}\n")

    # print("参数的数据类型已经输出到 'model_parameter_dtypes.txt' 文件中")


    if training_args.resume_path is None:
        logger.info('Training from scratch')
    else:
        logger.info('Traing from checkpoint: %s', training_args.resume_path)
        model.load_state_dict(torch.load(training_args.resume_path, map_location='cpu'))

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False
    

    if not model_args.train_vision_tower:
        if model_args.custom_train_vision_tower is not None:
            logger.info('You can not require not training vision tower but ask to train specific layers!') 
            return
        for k, v in model.named_parameters():
            if "model_clipv" in k or "model_visual" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False
    else:
        if model_args.custom_train_vision_tower is None:
            pass
        else:
            train_num = model_args.custom_train_vision_tower
            freeze_num = 12 - train_num
            freeze_layers = []
            for _i in range(freeze_num):
                layer_name = "model_visual.visual.blocks." + str(_i) + "."
                freeze_layers.append(layer_name)
            for k, v in model.named_parameters():
                if any(layer_name in k for layer_name in freeze_layers):
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False 

    # if not model_args.train_text_tower:
    #     for k, v in model.named_parameters():
    #         if "bge_encoder" in k or "bge_embeddings" in k or "bge_pooler" in k:
    #             # logging.info(f"Freeze the parameters for {k}")
    #             v.requires_grad = False 

    # skip_keywords = ['vision', 'text', 'depth', 'thermal', 'imu']

    # for name, param in model.named_parameters():
    #     if any(skip in name for skip in skip_keywords):
    #         param.requires_grad = False  # 冻结这部分参数
    # 冻结语言相关的参数
    for name, param in model.named_parameters():
        if "language" in name:
            param.requires_grad = False
            # logger.info(f"Freeze the parameters for {name}")


    # def set_audio_dropout(model, p=0.1):
    #     for name, module in model.named_modules():
    #         if "modality_trunks.audio" in name and isinstance(module, nn.Dropout):
    #             if module.p == 0.0:
    #                 # 替换为新的 Dropout
    #                 parent = get_parent_module(model, name)
    #                 attr_name = name.split('.')[-1]
    #                 setattr(parent, attr_name, nn.Dropout(p=p))
    #                 # print(f"Updated {name} to Dropout(p={p})")

    # def get_parent_module(model, module_name):
    #     parts = module_name.split('.')
    #     for part in parts[:-1]:
    #         model = getattr(model, part)
    #     return model
    # set_audio_dropout(model, p=0.2)  # 你可以调成你想要的 dropout 比例


    with open("trainable_parameters.txt", "w") as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"{name}: {param.shape}\n")

    with open("dropout_layers.txt", "w") as f:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                f.write(f"{name}: {module}\n")


    train_dataset = Multimodal_Dataset(args=data_args, 
                                    #    image_processor=model.preprocess_train,
                                      )
    # print(f"Config type1: {type(model.a_model.modality_config['audio'])}")
    # print(f"Config type2: {type(model.config)}")
    trainer = PreTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=Multimodal_Collator(
            model.a_model.modality_config['audio'],
            tokenizer,
        ),
        tokenizer=tokenizer,
        callbacks=[
        SaveBestCheckpointCallback(save_dir="/data1/ls/best_model", loss_threshold=0.3)
    ],
    )
    # if training_args.do_lr_finder:
    #     import matplotlib.pyplot as plt

    #     lrs = []
    #     losses = []

    #     # 1️⃣ 先手动初始化optimizer
    #     trainer.create_optimizer()

    #     # 2️⃣ 正常做LR扫描
    #     start_lr = 1e-7
    #     end_lr = 1
    #     num_steps = 100
    #     lr_mult = (end_lr / start_lr) ** (1/num_steps)

    #     for param_group in trainer.optimizer.param_groups:
    #         param_group['lr'] = start_lr
    #     device = torch.device(f"cuda:{training_args.local_rank}") if training_args.local_rank != -1 else torch.device("cuda")
    #     model = model.to(device)

    #     model.train()
    #     iter_count = 0

    #     def move_to_cuda(batch, device=None, fp16=False):
    #         if torch.is_tensor(batch):
    #             batch = batch.cuda() if device is None else batch.to(device)
    #             if fp16 and batch.dtype == torch.float32:
    #                 batch = batch.half()  # 转成float16
    #             return batch
    #         elif isinstance(batch, dict) or isinstance(batch, BatchEncoding):
    #             return {k: move_to_cuda(v, device, fp16) for k, v in batch.items()}
    #         elif isinstance(batch, (list, tuple)):
    #             return [move_to_cuda(v, device, fp16) for v in batch]
    #         else:
    #             return batch


       

    #     for batch in trainer.get_train_dataloader():
    #         batch = move_to_cuda(batch, device, fp16=training_args.fp16)


    #         outputs = model(**batch)
    #         loss = outputs.loss

    #         loss.backward()
    #         trainer.optimizer.step()
    #         trainer.optimizer.zero_grad()

    #         current_lr = trainer.optimizer.param_groups[0]['lr']
    #         lrs.append(current_lr)
    #         losses.append(loss.item())

    #         for param_group in trainer.optimizer.param_groups:
    #             param_group['lr'] = param_group['lr'] * lr_mult

    #         iter_count += 1
    #         if iter_count >= num_steps:
    #             break

    #     # 画图
    #     plt.plot(lrs, losses)
    #     plt.xscale('log')
    #     plt.xlabel('Learning Rate (log scale)')
    #     plt.ylabel('Loss')
    #     plt.title('Learning Rate Range Test')
    #     plt.grid()
    #     plt.savefig("/home/ls/FlagEmbedding/research/visual_bge/visual_bge/lr_finder_curve.png")
    #     print("Learning Rate Range Test done. Saved lr_finder_curve.png.")
    #     exit()

    # print("w2222",trainer.optimizer)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
    