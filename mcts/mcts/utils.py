import json
import re
import subprocess
import time
import requests
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import math
import argparse
from tqdm import tqdm 
import concurrent.futures
import torch
import numpy as np
import random
from vllm import LLM
import ast
import os
import warnings

EVAL_SYSTEM = 'You are a helpful assistant tasked with evaluating the quality of responses generated for function calling tasks.'
EVAL_PROMPT_with_GT = (
    'Ground Truth Response: {gt_ans}\n'
    'Generated Response by Model: {response}\n'
    
    'User Instruction:'
    'Please assess the quality of the generated response relative to the ground truth response.\n'
    'Note: A generated response that is a fragment of the ground truth response is also excellent.\n'

    'Evaluation Criteria:\n'
    '1. Function Name: Is the name of all the function called correct?\n'
    '2. Parameter Count: Is the number of parameters for all the function correct?\n'
    '3. Parameter Names: Are the names of all the parameters for the function correct?\n'
    '4. Parameter Value/Types: Are the value of all the parameters for the function correct?\n'
    '5. Semantic Similarity: Is the generated response semantically close to the ground truth response?\n'
    
    'Please directly choose from the following options to judge the overall quality:\n'
    '(A) Excellent: The generated response meets all criteria and is almost identical to the ground truth response.\n'
    '(B) Acceptable: The generated response meets most criteria but has minor discrepancies.\n'
    '(C) Fair: The generated response meets some criteria but has significant issues.\n'
    '(D) Poor: The generated response fails to meet most or all criteria.\n'

    'ASSISTANT: The option of overall quality is'
)

EVAL_PROMPT_without_GT = (
    "Question Input: {question}\n"
    "Generated Response(Completion) by Model: {response}\n"

    "User Instruction:\n"
    "Please assess the quality of the generated response.\n"

    "Evaluation Criteria:\n"
    "1. Is the name of the function called correct?\n"
    "2. Is the number of parameters for the function correct?\n"
    "3. Are the names of the parameters for the function correct?\n"
    "4. Are the values (types) of the parameters for the function correct?\n"
    "5. Can the input question be answered (e.g., are the necessary function  and parameters provided)?\n"

    "Please directly choose from the following options to judge the overall quality:\n"

    "(A) Excellent: The generated response meets all criteria.\n"
    "(B) Acceptable: The generated response meets most criteria but has minor discrepancies.\n"
    "(C) Fair: The generated response meets some criteria but has significant issues.\n"
    "(D) Poor: The generated response fails to meet most or all criteria.\n"

    "ASSISTANT: The option of overall quality is"
)

# Dataset and model loading functions
def init_dataset(dataset_path: str, tokenizer=None) -> List:
    # 1. Load JSON dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 2. Convert data to the desired format and store in a list
    dataset_list = []
    for conversation in json_data:
        system = conversation.get('system', '')
        dialogues = conversation.get('conversations', [])
        # Format conversation into a suitable template
#         formatted_conversations = [{"from": 'system', "value": system}]
        formatted_conversations = [{"role": 'system', "content": system}]
        answer = None  # Initialize answer field
        for i, turn in enumerate(dialogues):
            role = turn['from']
            content = turn['value']
            if i == len(dialogues) - 1:
                assert role == 'assistant', f'{dialogues}'
                answer = content
                continue
            formatted_conversations.append({
                "role": role,
                "content": content
            })
            if role == "tool": tool=1
            
        if formatted_conversations[-1]['role'] != 'user':
            continue
        
        full_conversation = tokenizer.apply_chat_template(conversation=formatted_conversations, tokenize=False, add_generation_prompt=True)
        dataset_list.append((full_conversation, answer))

    return dataset_list, json_data

# 随机种子固定
def set_seed(seed):
    # 设置PyTorch相关的随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 设置其他库的随机种子
    np.random.seed(seed)
    random.seed(seed)

class VLLMServer:
    def __init__(self, model_path: str, num_gpus: int, gpu_memory_utilization: float = 0.9, vllm_port: int = 8001):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.gpu_memory_utilization = gpu_memory_utilization
        self.vllm_port = vllm_port
        self.process = None
        self.max_model_len = 8196

    def start(self):
        command = [
            "vllm", 
            "serve", self.model_path, 
            "--port", str(self.vllm_port),
            "--dtype", "float16",
            "--tensor-parallel-size", str(self.num_gpus),
            "--max_model_len", str(self.max_model_len), 
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--uvicorn-log-level", "warning",
            "--trust-remote-code",
            "--disable-log-requests",
        ]
        # todo add vllm.llm init
        
        try:
            self.process = subprocess.Popen(command)
            server_ready = False
            while not server_ready:
                if self.process.poll() is not None:
                    raise Exception(f"Subprocess terminated unexpectedly with code {self.process.returncode}")
                try:
                    response = requests.get(f"http://localhost:{self.vllm_port}/v1/models")
                    if response.status_code == 200:
                        server_ready = True
                        print("Server is ready!")
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
        except Exception as e:
            print(f"Failed to start server: {e}")
            if self.process:
                self.stop()
            raise
        
    def get_client(self):
        #return llm, sampling, tokenizer
        client = OpenAI(base_url=f"http://localhost:{self.vllm_port}/v1", api_key="EMPTY")
        models = client.models.list()
        model = models.data[0].id
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left', model_max_length=self.max_model_len)
        return client, model, tokenizer

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=15)
                print("Process terminated successfully.")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                print("Process killed due to timeout.")
            finally:
                self.process = None

def process_item(client, model_name, prompt, temperature, max_tokens, seed):
    try:
        output = client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            # n=1,
            logprobs=1,
            seed=seed,
            extra_body={
                "repetition_penalty": 0.75,
                "skip_special_tokens": False,
            }
        )
        choices = output.choices[0]
        token_logprobs = choices.logprobs.token_logprobs if choices.logprobs else []
        avg_prob = math.exp(sum(token_logprobs) / len(token_logprobs)) if token_logprobs else None
        result_text = choices.text.strip()
        return {
            'generated_response': result_text,
            'prob_conf': avg_prob
        }
    except Exception as e:
        print(f"Error processing prompt: {prompt}. Error: {e}")
        return None

def infer_and_save_buffer(server: VLLMServer, dataset_path: str, max_tokens: int, temperature: float, seed: int, max_workers: int = 10, iter=1):
    client, model_name, tokenizer = server.get_client()
    dataset_list, raw_data = init_dataset(dataset_path, tokenizer)
    
    # Prepare the arguments for each thread
    futures_to_index = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, ((prompt, _), data_item) in enumerate(zip(dataset_list, raw_data)):
            future = executor.submit(
                process_item,
                client, model_name, prompt, temperature, max_tokens, seed
            )
            futures_to_index[future] = (index, data_item)

        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(futures_to_index), total=len(futures_to_index), desc='Getting prob conf:'):
            index, data_item = futures_to_index[future]
            result = future.result()
            if result is not None:
                data_item.update(result)

    # Save the results to the output file
    output_path = dataset_path.replace('.json',f'_{iter}_infer_buffer.json')
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)

    return raw_data

def parse_arguments():
    parser = argparse.ArgumentParser(description="VLLM Server with Dataset Inference")
    parser.add_argument("--model_path", type=str, required=True, default='path_to_model', help="Path to the train model")
    parser.add_argument("--infer_dataset_path", type=str, default="path_to_save/dataset_path.json", help="Path to the train dataset") 
    parser.add_argument('--save_buffer_name', type=str, default='path_to_save/buffer_ppl.json', help='Directory to temp save the ppl buffer data')
    parser.add_argument('--save_preference_dir', type=str, default='path_to_save/preference.json', help='Directory to save the preference data')
    parser.add_argument('--output_dir', type=str, default='/path_to_save/models/save_model_path', help='Directory to save the models')
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs or NPU to use")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--max_model_tokens", type=int, default=8192, help="Maximum number of tokens to model input")
    parser.add_argument("--num_actions", type=int, default=2, help="Number of actions to perform")
    parser.add_argument("--temperature", type=float, default=1.5, help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--step_limit", type=int, default=1, help="Max sentence/tools to consist of a step")
    parser.add_argument("--depth", type=int, default=2, help="Max sentence to consist of a step")
    parser.add_argument('--max_simulation_steps', type=int, default=2, help='Maximum number of simulation steps')
    # parser.add_argument("--repetition_penalty", type=float, default=0.75, help="Repetition penalty")
    #intra args
    parser.add_argument("--actor_model", type=str, default=None, help="actor model client")
    parser.add_argument("--model_name", type=str, default=None, help="model name to client")
    parser.add_argument('--iter', type=int, default=1, help='Current iterative of mcts')

    # 添加新的超参数
    parser.add_argument('--n_workers', type=int, default=30, help='Number of worker threads for processing')
    parser.add_argument('--dpo_iters', type=int, default=3, help='Number of iterative for all dpo')
    parser.add_argument('--mcts_iters', type=int, default=5, help='Number of iterative for mcts')
    parser.add_argument('--exploration_weight', type=float, default = 1.0, help='Exploration Weight of select in MCTS')
    parser.add_argument('--value_threshold', type=float, default = 0.1, help='偏好数据差值最小值')
    parser.add_argument('--accept_value', type=float, default = -0.3, help='偏好数据中accept 样本的最小value 值')
    parser.add_argument('--overlap_limit', type=float, default = 0.95, help='去重值')
    parser.add_argument('--ratio', type=float, default = 0.3, help='主动学习采样比例')
    parser.add_argument('--pref_loss', type=str, default = 'sigmoid', help='type of prefer learning')
    parser.add_argument('--pref_beta', type=float, default = 0.01, help='type of prefer learning')
    parser.add_argument('--simpo_gamma', type=float, default = 1.4, help='type of prefer learning')
    parser.add_argument('--prefer_lr', type=float, default = 1e-6, help='type of prefer learning')

    return parser.parse_args()


def cal_ppl(args):    
    base_name = os.path.basename(args.infer_dataset_path)[:-5]
    dirname = os.path.dirname(args.infer_dataset_path)
    command = [
        'torchrun', 
        f'--nproc_per_node={args.num_gpus}', 
        '--master_port=12346', 
        '--nnodes=1',
        '--node_rank=0',
        'cal_ppl.py',
        '--model_name_or_path', args.model_path,
        '--dataset', base_name,
        '--dataset_dir', dirname,
        '--template', 'llama3',
        '--save_name', args.save_buffer_name,
        '--batch_size', '1'
    ]
    script_dir = "./scripts"
    print(f'this is step 1, cal ppl,{script_dir}\n\n')
    result = subprocess.run(command, check=True, capture_output=False, text=True, cwd=script_dir )
    print(result)
    
def active_select(buffer_name=None, iter=1, ratio = 0.3):

    # 读取 JSON 文件中的数据
    with open(buffer_name, 'r', encoding="utf-8") as f:
        buffer = json.load(f)
        with_tool_data = []
        with_no_tool_data = []
        for item in buffer:
            if 'perplexity' in item.keys():
                if ')' in item['conversations'][-1]['value']:
                    with_tool_data.append(item)
                else:
                    with_no_tool_data.append(item)
    
    # 根据 perplexity 排序，从高到低，困难度越高的在前,前30%的数据数量
    with_tool_data = sorted(with_tool_data, key=lambda x: x['perplexity'], reverse=True)
    selected_with_tool_data = with_tool_data[:int(len(with_tool_data) * ratio)]
    with_no_tool_data = sorted(with_no_tool_data, key=lambda x: x['perplexity'], reverse=True)
    selected_with_no_tool_data = with_no_tool_data[:int(len(with_no_tool_data) * ratio)]
    
    selected_data = selected_with_tool_data + selected_with_no_tool_data
    
    for dictionary in selected_data: # 检查字典中是否包含 'perplexity' 键
        if 'perplexity' in dictionary: # 如果存在，则删除该键
            del dictionary['perplexity']
    print(f'step 2, len of buffer {len(buffer)}, len of num selected {len(selected_data)} !!')
    
    selected_buffer_name = buffer_name.replace('.json',f'_selected_{iter}.json').replace('mcts','data')
    # 将选中的数据保存到新的文件
    with open(selected_buffer_name, 'w', encoding="utf-8") as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=4)

    return selected_buffer_name



def enhance_update_data(input_path, save_path = None, iter = 1):
    ENHANCE_PROMPT = ''' 
        You are a professional data imitator. I will provide you with a sample of a tool invocation, formatted as a dictionary containing `system` and `conversations` fields. The sample is as follows:
        {sample}

        Based on this sample, please generate a new conversation with the following requirements:
        1. Keep the `system` content unchanged, imitate the structure of the `conversations` section, modify the user request, and adjust the parameter values in the tool invocation (while keeping the format unchanged).
        2. The roles in the conversation should be the same as in the sample.
        3. When the `assistant` invokes the tool, the parameters must conform to the tool's defined format.
        4. The language style should be consistent with the sample.
        5. The conversation must start with `user` and end with `assistant`.

        The output format should only include the `conversations` list, which must be complete and comply with the JSON Schema.
    '''
    url = "https://api.openai-sb.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sb-xx",
        "Content-Type": "application/json",
    }
    warnings.filterwarnings("ignore")
    enhanced_data = []
    def process_item(item, idx):
        """处理单个项的函数"""
        sample_str = ENHANCE_PROMPT.format(sample= json.dumps(item, ensure_ascii=False) )
        # print(len(sample_str),sample_str)
        request_body = {
            "messages": [{'role': 'user', 'content': sample_str}],  # 使用样本生成的字符串
            "max_tokens": 8192,  # 设置最大长度
            "model": "gpt-4o-mini-2024-07-18", #gpt-4o-2024-11-20, gpt-4o-mini-2024-07-18
        }
        # print('this raw : ',item['conversations'])
        try:
            # chat_response = requests.post(url=u, headers=h, json=request_body, verify=False, proxies=proxies)
            chat_response = requests.post(url, headers=headers, json=request_body, verify=False)
            if chat_response.status_code == 200:
                r = chat_response.json()
                res = r["choices"][0]["message"]["content"]  # 获取响应内容
                return {'id': idx, 'system': item['system'], 'conversations': res}
            else:
                return f'Error: {chat_response.status_code}, {chat_response.text}'
        except requests.exceptions.RequestException as e:
            return f'Error: {e}'


    # 验证工具调用语法
    def validate_tool_call(value):
        try:
            # 尝试解析工具调用字符串
            parsed_ast = ast.parse(value.strip(), mode='eval')  # 'eval' 解析单个表达式
            if isinstance(parsed_ast.body, ast.List):
                for elt in parsed_ast.body.elts:
                    if not isinstance(elt, ast.Call):
                        return False  # 如果解析为函数调用并且参数有效，返回 True
                return True
        except SyntaxError as e:
            print(f"Syntax Error: {e}")
            return False  # 解析失败，说明不是有效的工具调用语法

        return False  # 不是有效的函数调用

    # 检查数据的主要函数
    def check_data(data_list):
        res_list = []
        for item in data_list:
            try:
                conversations = item['conversations'].strip()
                if '```' in conversations: #考虑```json {}``` 情况
                    conversations = conversations[7:-3].strip()
                if conversations.startswith('"conversations"'): #考虑conversations：[] 情况 
                    conversations = '{' + conversations + '}'
                # Step 1: 尝试将字符串解析为 JSON 格式，判断是否是有效的 JSON
                json_data = json.loads(conversations)
                if type(json_data) is dict and type(json_data['conversations']) is list: #考虑{考虑conversations：[]} 情况
                    json_data = json_data['conversations']
                item['conversations'] = json_data
            except Exception as e:
                res_list.append([])  # 如果 JSON 解析失败，返回空列表
                continue

            
            try:
                # Step 2: 遍历 conversations 中的每一项
                # print(item['conversations'])
                for idx, each in enumerate(item['conversations']):
                    if idx % 2 == 0:  # 用户或工具的发言
                        if each['from'] in ['user', 'tool'] and len(each['value']) > 0:
                            pass  # 用户和工具的发言没有问题
                        else:
                            raise ValueError('Value error in each user/tools')
                    else:  # assistant 的发言
                        if each['from'] == 'assistant':
                            # 如果 value 中包含 '['，则可能是工具调用，使用 AST 验证
                            if '[' in each['value']:
                                if not validate_tool_call(each['value']):
                                    pass
                                    # raise ValueError(f"Invalid tool call syntax: {each['value']}")
                            elif len(each['value']) > 0:
                                pass  # assistant 的发言内容不为空也通过
                            else:
                                raise ValueError('Value error in each assistant')
                
                # Step 3: 答案是否正确（此步骤可以根据需要使用 vllm 进行判断）
                res_list.append(item)  # 如果没有抛出异常，表示数据有效，添加到结果列表
            except Exception as e:
                print(str(e))
                # traceback.print_exc()  # 打印堆栈跟踪
                res_list.append([])  # 出现错误时，返回空列表

        return res_list

    with open(input_path, 'r', encoding="utf-8") as f:
        buffer = json.load(f)[8000:] # 4000,[4000,8000), [8000, all)
        # 使用 ThreadPoolExecutor 来并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [(executor.submit(process_item, item, idx + 4000)) for idx, item in enumerate(buffer) ]
            
            # 等待任务完成并收集结果
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if isinstance(result, dict):  # 正常返回的结果
                    enhanced_data.append(result)
                else:  # 错误信息
                    enhanced_data.append([])
                    print('this is error',result)

    # print(len(enhanced_data) )
    checked_data = check_data(enhanced_data)
    enhanced_path = input_path.replace('.json', f'_enhanced_{iter}.json')
    with open(enhanced_path, 'w', encoding="utf-8") as f:
        json.dump(checked_data, f, ensure_ascii=False, indent=2)

    # update save path
    # with open(save_path, 'w+', encoding="utf-8") as f:
    #     save_buffer = json.load(f)
    #     save_buffer.extend(checked_data)
    #     json.dump(save_buffer, f, ensure_ascii=False, indent=2)

    # 先读再写，保存enhanced data的内容
    with open(input_path, 'r', encoding="utf-8") as f:
        save_buffer = json.load(f)
        save_buffer.extend(checked_data)
    # with open(input_path, 'w', encoding="utf-8") as f:
    #     json.dump(save_buffer, f, ensure_ascii=False, indent=2)
    #     return save_buffer


if __name__ == "__main__":
    pass
