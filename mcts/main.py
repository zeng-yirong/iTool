
# adapter from https://github.com/maitrix-org/llm-reasoners/blob/main/reasoners/algorithm/mcts.py
# second from https://github.com/YuxiXie/MCTS-DPO/blob/main/mcts_rl/algorithms/mcts/main.py
from utils import set_seed, parse_arguments, enhance_update_data, cal_ppl, active_select
from tqdm import trange
from mcts_vllm import get_dpo_data_by_mcts
from dpo_train import dpo_train, sft_train
import time
import os 
import psutil

def find_and_kill_process_by_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.info['connections']:
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    print(f"Found process {proc.info['pid']} using port {port}. Terminating...")
                    proc.kill()
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    print(f"No process found using port {port}.")
    return False
    
if __name__ == '__main__':
    """Main training routine."""
    args = parse_arguments()
    set_seed(args.seed)

    for iter in trange(0, args.dpo_iters, desc = 'start itering:'):
        args.iter=iter
        # step1 加入数据，初始非偏好数据，后面还有偏好数据
        cal_ppl(args=args) 
        selected_buffer_path = active_select(buffer_name=args.save_buffer_name, iter=args.iter, ratio = args.ratio)  # step 2 两种数据的选择
        # selected_buffer_path = enhance_update_data(selected_buffer_path, args.infer_dataset_path, iter=iter) #step 3 gpt4o 进行增强，更新到original data 中
        # 考虑此处添加sft 回放策略
        # args.model_path = sft_train(args=args, dataset = selected_buffer_path)
        get_dpo_data_by_mcts(args=args, dataset_path=selected_buffer_path) #step 3
        # dpo training
        args.model_path = dpo_train(args=args, learning_rate = args.prefer_lr, pref_loss = args.pref_loss, pref_beta = args.pref_beta, simpo_gamma = args.simpo_gamma) #step 4 偏好训练
        time.sleep(60)
        os.system('npu-smi info')
        
    print(args.model_path)
