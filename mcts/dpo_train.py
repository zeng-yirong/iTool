import subprocess
import os

def dpo_train(
    args=None,
#     model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    #qlora
    quantization_bit: int =  8,
    quantization_method: str = 'bitsandbytes',  # choices: [bitsandbytes (4/8), hqq (2/3/4/5/6/8), eetq (8)]

    stage: str = "dpo",
    do_train: bool = True,
    finetuning_type: str = "lora",
    lora_target: str = "all",
    lora_rank: int = 8,
    # special lora
    use_rslora: bool = False,
    use_dora: bool = False,
    loraplus_lr_ratio: float = 16.0,
    simpo_gamma: float = 0.5,
    pref_beta: float = 0.1,
    pref_loss: str = "sigmoid", # sigmoid (dpo), orpo, simpo
    
#     dataset: str = "dpo_en_demo",
    max_samples: int = "4000",
    template: str = "llama3",
    cutoff_len: int = 2048,
    overwrite_cache: bool = True,
    preprocessing_num_workers: int = 16,
#     output_dir: str = "saves/llama3-8b/lora/dpo",
    logging_steps: int = 50,
    save_steps: int = 40,
    plot_loss: bool = True,
    overwrite_output_dir: bool = True,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5.0e-6, #5e-6->2e-6->1e-6
    num_train_epochs: float = 2.0,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    bf16: bool = True,
    fp16: bool = False,
    ddp_timeout: int = 180000000,
    val_size: float = 0.01,
    per_device_eval_batch_size: int = 1,
    eval_strategy: str = "steps",
    eval_steps: int = 40,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = 'loss',
    greater_is_better: bool = False
):
#     learning_rate = round(1/(args.iter+1),2)*1e-6

    output_dir = args.output_dir + f'_{args.iter}'
    dataset = os.path.basename(args.save_preference_dir)[:-5] + f'_{args.iter}'
    # 构建命令行参数列表, train
    cmd = [
        "llamafactory-cli", "train",
        "--model_name_or_path", args.model_path,
        # "--quantization_bit", str(quantization_bit),
        # "--quantization_method", quantization_method,
        
        "--stage", stage,
        "--do_train", str(do_train).lower(),
        "--finetuning_type", finetuning_type,
        "--lora_target", lora_target,
        "--lora_rank", str(lora_rank),
        
        "--use_dora", str(use_dora).lower(),
        "--pref_beta", str(pref_beta),
        "--pref_loss", pref_loss,
        "--simpo_gamma", str(simpo_gamma),
        
        "--max_samples", str(max_samples),
        "--dataset", dataset,
        "--template", template,
        "--cutoff_len", str(cutoff_len),
        "--overwrite_cache", str(overwrite_cache).lower(),
        "--preprocessing_num_workers", str(preprocessing_num_workers),
        "--output_dir", output_dir,
        "--logging_steps", str(logging_steps),
        "--save_steps", str(save_steps),
        "--plot_loss", str(plot_loss).lower(),
        "--overwrite_output_dir", str(overwrite_output_dir).lower(),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(num_train_epochs),
        "--lr_scheduler_type", lr_scheduler_type,
        "--warmup_ratio", str(warmup_ratio),
        "--bf16", str(bf16).lower(),
        "--fp16", str(fp16).lower(),
        "--ddp_timeout", str(ddp_timeout),
        # "--val_size", str(val_size),
        # "--per_device_eval_batch_size", str(per_device_eval_batch_size),
        # "--eval_strategy", eval_strategy,
        # "--eval_steps", str(eval_steps),
        # "--save_total_limit", str(save_total_limit),
        # "--load_best_model_at_end", str(load_best_model_at_end).lower(),
        # "--metric_for_best_model", str(metric_for_best_model),
        # "--greater_is_better", str(greater_is_better).lower()
    ]

    # 调用 subprocess 执行命令
    script_dir = '/opt/huawei/dataset/data_dir/zyr/LLaMA-Factory-main/'
    print('this dpo trainning !', script_dir)
    res = subprocess.run(cmd, check=True, capture_output=False, text=True, cwd=script_dir)
    print(res)
    # merge
    export_dir = output_dir.replace('saves','merged')
    cmd = [
        "llamafactory-cli", "export", 
        f"--model_name_or_path={args.model_path}",
        f"--adapter_name_or_path={output_dir}",
        f"--template={template}",
        f"--finetuning_type={finetuning_type}",
        f"--export_dir={export_dir}",
        f"--export_size={5}",
        f"--export_device={'cpu'}",
        f"--export_legacy_format={False}",
    ]
    res = subprocess.run(cmd, check=True, capture_output=False, text=True, cwd=script_dir )
    
    print(res)
    return export_dir

def sft_train(
    args=None,
#     model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",

    stage: str = "sft",
    do_train: bool = True,
    finetuning_type: str = "lora",
    lora_target: str = "all",
    lora_rank: int = 16,
    # special lora
    use_rslora: bool = False,
    use_dora: bool = True,
    loraplus_lr_ratio: float = 16.0,

    dataset: str = "dpo_en_demo",
    template: str = "llama3",
    cutoff_len: int = 4096,
    overwrite_cache: bool = True,
    preprocessing_num_workers: int = 16,
#     output_dir: str = "saves/llama3-8b/lora/dpo",
    logging_steps: int = 20,
    save_steps: int = 50,
    plot_loss: bool = True,
    overwrite_output_dir: bool = True,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1.0e-6, #5e-6->2e-6->1e-6
    num_train_epochs: float = 2.0,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    bf16: bool = False,
    fp16: bool = True,
    ddp_timeout: int = 180000000,
    val_size: float = 0.05,
    per_device_eval_batch_size: int = 1,
    eval_strategy: str = "steps",
    eval_steps: int = 20,
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = 'loss',
    greater_is_better: bool = False
):
    
    output_dir = args.model_path.replace('merged', 'saves') + '_sft'
    dataset = os.path.basename(dataset)[:-5]
    # 构建命令行参数列表, train
    cmd = [
        "llamafactory-cli", "train",
        "--model_name_or_path", args.model_path,
        
        "--stage", stage,
        "--do_train", str(do_train).lower(),
        "--finetuning_type", finetuning_type,
        "--lora_target", lora_target,
        "--lora_rank", str(lora_rank),
        
        "--use_rslora", str(use_rslora).lower(),
        "--use_dora", str(use_dora).lower(),
        
        "--dataset", dataset,
        "--template", template,
        "--cutoff_len", str(cutoff_len),
        "--overwrite_cache", str(overwrite_cache).lower(),
        "--preprocessing_num_workers", str(preprocessing_num_workers),
        "--output_dir", output_dir,
        "--logging_steps", str(logging_steps),
        "--save_steps", str(save_steps),
        "--plot_loss", str(plot_loss).lower(),
        "--overwrite_output_dir", str(overwrite_output_dir).lower(),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(num_train_epochs),
        "--lr_scheduler_type", lr_scheduler_type,
        "--warmup_ratio", str(warmup_ratio),
        "--bf16", str(bf16).lower(),
        "--fp16", str(fp16).lower(),
        "--ddp_timeout", str(ddp_timeout),
#         "--val_size", str(val_size),
#         "--per_device_eval_batch_size", str(per_device_eval_batch_size),
#         "--eval_strategy", eval_strategy,
#         "--eval_steps", str(eval_steps),
#         "--save_total_limit", str(save_total_limit),
#         "--load_best_model_at_end", str(load_best_model_at_end) 
#         "--metric_for_best_model", "loss",
#         "--greater_is_better", str(True),
    ]

    # 调用 subprocess 执行命令
    script_dir = '/opt/huawei/dataset/data_dir/zyr/LLaMA-Factory-main/'    
    print('this replay sft trainning !', script_dir)
    res = subprocess.run(cmd, check=True, capture_output=False, text=True, cwd=script_dir)
#     print(res)
    # merge
    export_dir = output_dir.replace('saves','merged')
    cmd = [
        "llamafactory-cli", "export", 
        f"--model_name_or_path={args.model_path}",
        f"--adapter_name_or_path= {output_dir}",
        f"--template={template}",
        f"--finetuning_type={finetuning_type}",
        f"--export_dir={export_dir}",
        f"--export_size={5}",
        f"--export_device={'cpu'}",
        f"--export_legacy_format={False}",
    ]
    res = subprocess.run(cmd, check=True, capture_output=False, text=True, cwd=script_dir )
    return export_dir

if __name__ == "__main__":
    dpo_train()
