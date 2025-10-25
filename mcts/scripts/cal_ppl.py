# coding=utf-8
# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer

# from accelerate import Accelerator
# import torch
import os
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn
from torch.utils.data import Sampler
import math
# from accelerate import Accelerator

@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    train_on_prompt: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        chosen_features = []
        for feature in features:
            prompt_len, answer_len = len(feature["prompt_ids"]), len(feature["chosen_ids"])
            input_ids = feature["prompt_ids"] + feature["chosen_ids"]
            attention_mask = [1] * (prompt_len + answer_len)
            labels = input_ids if self.train_on_prompt else [IGNORE_INDEX] * prompt_len + feature["chosen_ids"]
            chosen_features.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

        return super().__call__(chosen_features)

def calculate_ppl(
    model_name_or_path: str,
    save_name: str,
    batch_size: int = 1,
    stage: Literal["pt", "sft", "rm"] = "sft",
    dataset: str = "plugin_30_percent_splited",
    dataset_dir: str = "./LLaMA-Factory-main/data",
    template: str = "llama3",
    cutoff_len: int = 8192,
    max_samples: Optional[int] = None,
    train_on_prompt: bool = False,
):
    r"""
    Calculates the ppl on the dataset of the pre-trained models.
    Usage: python cal_ppl.py --model_name_or_path path_to_model --dataset alpaca_en_demo --save_name ppl.json
    accelerate launch --num_processes=8 --main_process_port 41011 
    python cal_ppl.py --model_name_or_path path --dataset plugin_30_percent_splited --template llama3 --save_name path/buffer_ppl.json
    """

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # Set the device to the local GPU
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Initialize model and tokenizer arguments
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        dict(
            stage=stage,
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            train_on_prompt=train_on_prompt,
            output_dir="dummy_dir",
            overwrite_cache=True,
            do_train=True,
        )
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    trainset = get_dataset(template, model_args, data_args, training_args, stage, **tokenizer_module)["train_dataset"]
    trainset = trainset.remove_columns(['images', 'videos'])
    
    # 定义添加 `id` 列的函数
    def add_id(example, idx):
        example['id'] = idx
        return example
    trainset = trainset.map(add_id, with_indices=True)
        
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    
    # Set up the model for distributed training
    model = model.to(device)
#     model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Setup data collator based on the stage
    if stage == "pt":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif stage == "sft":
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
    elif stage == "rm":
        data_collator = PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX, train_on_prompt=train_on_prompt
        )
    else:
        raise NotImplementedError("Stage does not supported: {}.".format(stage))

    # Create DistributedSampler for the dataset
    train_sampler = DistributedSampler(
        trainset, 
        drop_last=True,
        shuffle=False
    )
    
    # Create DataLoader with the DistributedSampler
    dataloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=data_collator, 
        pin_memory=True, 
        sampler=train_sampler,
        drop_last = True
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    total_ppl = 0
    perplexities = []
    idx_list = []

    model.eval()
            
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc = 'Getting Buffer PPL: ' ):
            cur_idx_list = batch['id'].tolist()
            batch = {k: v.to(device) for k, v in batch.items() if k != 'id'}
            outputs = model(**batch)
            shift_logits = outputs["logits"][..., :-1, :]
            shift_labels = batch["labels"][..., 1:]
            loss_mask = shift_labels != IGNORE_INDEX
            
            flatten_logits = shift_logits.contiguous().view(shift_labels.size(0) * shift_labels.size(1), -1)
            flatten_labels = shift_labels.contiguous().view(-1)
            
            token_logps = criterion(flatten_logits, flatten_labels)
            token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)
            
            sentence_logps = (token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            total_ppl += sentence_logps.exp().sum().item()
            perplexities.extend(sentence_logps.exp().tolist())
            idx_list.extend(cur_idx_list)

        def gather_results(results, world_size, device, dtype = torch.float32):
            results = torch.tensor(results, dtype=dtype, device=device)
            gathered_results = [torch.zeros_like(results) for _ in range(world_size)]
            dist.all_gather(gathered_results, results)
            return gathered_results

        gathered_perplexities = gather_results(perplexities, world_size, device=device)
        gathered_idx_list = gather_results(idx_list, world_size, device=device, dtype = torch.int64)        
        # Save the merged data to the specified file
        if dist.get_rank() == 0:  # Only save from the main process
            all_perplexities = []
            all_idx = []
            for rank in range(world_size):
                all_perplexities.extend(gathered_perplexities[rank].tolist())
                all_idx.extend(gathered_idx_list[rank].tolist())
            
            # Ensure we have the same number of samples in both lists
            original_data_path = f"{dataset_dir}/{dataset}.json"
            with open(original_data_path, 'r', encoding="utf-8") as f:
                original_data: List[Dict[str, Any]] = json.load(f)

            print('the len of perplexities: ',len(all_idx), len(all_perplexities), len(original_data), len(trainset), all_perplexities[:1], sep='\n')
            with open('ppl.json', "w", encoding="utf-8") as f:
                json.dump(all_perplexities + all_idx, f, indent=2, ensure_ascii=False)
            
            for idx, perplexity in zip(all_idx, all_perplexities):
                try:
                    original_data[int(idx)]['perplexity'] = perplexity
                except Exception as e:
                    print(f"An error occurred: {idx}, {str(e)}")
            with open(save_name, "w", encoding="utf-8") as f:
                json.dump(original_data, f, indent=2, ensure_ascii=False)

#             print(f"Average perplexity on {dist.get_rank()} is {total_ppl / len(perplexities):.2f}")
            print(f"Perplexities have been saved at {save_name}.")
    
    # Clean up the distributed environment
    dist.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(calculate_ppl)
    # 启动DDP
    # fire.Fire(calculate_ppl_ddp)