import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
import numpy as np
import torch
import math
import json
import weakref
import re
from utils import EVAL_PROMPT_with_GT, EVAL_PROMPT_without_GT, EVAL_SYSTEM
from utils import VLLMServer,init_dataset
import random

class State:
    def __init__(self, all_prompt='Where is Beijing?', last_response='', stop=None):
        self.all_prompt = all_prompt #也可以考虑使用一个path 存储 history node
        self.last_response = last_response
        self.stop = stop

class Node:
    def __init__(self, state, parent=None, gt_ans=None, prob=0, args=None):
        self.state = state
        self.parent = weakref.ref(parent) if parent is not None else None
        self.children = []
        self.visits = 0
        self.prob = prob
        self.gt_ans = gt_ans
        self.value = 0  # Initial value for the node
        self.depth = 0 if parent is None else parent.depth + 1
        self.args = args

    def best_child(self, exploration_weight=1.0):
        choices_weights = [
            (child.value / (child.visits + 1)) + exploration_weight * child.prob * np.sqrt(np.log(self.visits) / (child.visits + 1))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        legal_actions = get_actions(self.state, args=self.args)  # Now returns a list of lists of actions
        def expand_recursive(current_node, actions_list):
            if not actions_list:
#                 print('this is no actions_list', actions_list)
                return
            
            # Check if the depth is greater than 2, 
            if current_node.depth >= self.args.depth:
                # Combine remaining actions into a single action
                actions_list = [(' '.join(t[0] for t in actions_list), actions_list[-1][1], actions_list[-1][2])]
            
            current_action, stop, prob = actions_list[0]
            
            new_state = perform_action(current_node.state, current_action, stop=stop)
            child_node = Node(new_state, parent=current_node, gt_ans=current_node.gt_ans, prob=prob, args = self.args)
            
            # Use reward_model_fast to set an initial value (reward) for the child
            reward = get_reward(current_node.state, current_action, fast_reward=True, gt_ans=current_node.gt_ans, reward_model=self.args.actor_model, reward_model_name=self.args.model_name)
            child_node.value += reward
            current_node.children.append(child_node)
            
            # Recursively expand the next level of actions
            expand_recursive(child_node, actions_list[1:])
        
        for actions_list in legal_actions:
            expand_recursive(self, actions_list)
        
        return self.children

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent().backpropagate(reward)
    
    def is_terminal(self):
        """Checks if the state is a terminal state."""
        assert isinstance(self.state, State), "state must be an instance of State"
        sum_tokens_num = len(self.state.all_prompt.split())/4 + 1

        if sum_tokens_num >= self.args.max_model_tokens - 5:
            return True
        elif len(self.state.last_response) > 0 and self.state.stop:
            return True
        else:
            return False

def calculate_overlap_ratio(str1, str2): #通过重合率判断
    str1_set, str2_set = set(str1.split()), set(str2.split())
    intersection = str1_set.intersection(str2_set)
    overlap_ratio = len(intersection) / min(len(str1_set), len(str2_set))
    return overlap_ratio

@torch.no_grad()
def get_actions(state, break_down=True, args=None):
    """
    Given a state, the policy model suggests a list of possible actions.
    This can be a call to an LLM to determine the best actions given the current state.
    """
    sequences = args.actor_model.completions.create(
        model = args.model_name, # or model_path ?
        prompt = state.all_prompt,
        temperature=args.temperature,
        max_tokens= args.max_tokens,
        n=args.num_actions,
        logprobs = 1,
        seed=args.seed,
        extra_body={
            "repetition_penalty":0.75,
            "skip_special_tokens": False,
        }
    )
    generated_actions = [output for output in sequences.choices]

    if break_down:
        broken_down_actions = []
        for action in generated_actions:
            action_text = action.text
            stop = action.finish_reason == 'stop' 
            split_pattern = r'(\. |\, |=|\()' # 分割符
            sentences = re.split(split_pattern, action_text)
            # 合并分隔符和前一个字符串
            merged_result = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    merged_result.append(sentences[i] + sentences[i + 1])
                else:
                    merged_result.append(sentences[i])
            
            grouped_sentences = [(' '.join(merged_result[i:i+args.step_limit]),None,None) for i in range(0, len(merged_result), args.step_limit)]
            token_id = 0
            for idx, item in enumerate(grouped_sentences):
                cur_prob = 0
                cnt=0
                sent = item[0]
                for token, log_prob in zip(action.logprobs.tokens[token_id:], action.logprobs.token_logprobs[token_id:]):
                    if token.strip() not in sent:
                        break
                    cur_prob += log_prob 
                    cnt += 1
                    token_id += 1
                if cnt != 0: 
                    grouped_sentences[idx] = (grouped_sentences[idx][0], None, math.exp(cur_prob/cnt) )
                else:
                    grouped_sentences[idx] = (grouped_sentences[idx][0], None, 0.1)
            if len(grouped_sentences) == 0:
                broken_down_actions.append([])
                continue
            if stop:
                grouped_sentences[-1] = (grouped_sentences[-1][0], stop, grouped_sentences[-1][2]) #(text, stop, prob) 更新最后 一个值
            broken_down_actions.append(grouped_sentences) #添加所有的action candidates
        return broken_down_actions # 元组列表
    
    return [(action.text,action.finish_reason == 'stop' , sum(action.logprobs.token_logprobs) ) for action in generated_actions ] #一个补全的文本字串 列表

@torch.no_grad()
def get_reward(pre_state, action, fast_reward = False, gt_ans=None, reward_model=None, reward_model_name=None):
    if len(action.strip()) < 1 : return 0
    all_reward = 0
    
    # step 1 : 通过 启发工进行反馈模型, 是否正常识别出函数，后期可以使用auto-checker 进行判断反馈
    if gt_ans:
        if ( ')]' in action and ')]' in gt_ans) or ( ')]' not in action and ')]' not in gt_ans):
            all_reward += 0.1
        else:
            all_reward += -0.1
        reward = calculate_overlap_ratio(gt_ans, action)-0.5
        all_reward += reward
    else:
        all_reward += 0.1 if ')' in action else -0.1

    if fast_reward:
        return all_reward

    # step 2: self_eval 
    else:
        def self_eval(gt_ans=None, response=None, question_input = None, max_tokens=7, seed=42):
            if gt_ans:
                prompt = EVAL_SYSTEM + EVAL_PROMPT_with_GT.format(gt_ans = gt_ans, response=response)
            else:
                prompt = EVAL_SYSTEM + EVAL_PROMPT_without_GT.format(question = question_input, response=response)

            # 使用 vLLM 的 client.completions.create 获取补全结果
            completion = reward_model.completions.create(
                model=reward_model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.6,
                logprobs=4,
                seed=seed,
            ).choices[0]

            confs={'A':0,'B':0, 'C':0, 'D':0}
            for top_logprobs in completion.logprobs.top_logprobs:
                for token, logprob in top_logprobs.items():
                    if token.lower() in '(A)'.lower() or token.lower() in 'Excellent'.lower():
                        confs['A'] += math.exp(logprob)
                        break
                    if token.lower() in '(B)'.lower() or token.lower() in 'Acceptable'.lower():
                        confs['B'] += math.exp(logprob)
                        break

                    if token.lower() in '(C)'.lower() or token.lower() in 'Fair'.lower():
                        confs['C'] += math.exp(logprob)
                        break

                    if token.lower() in '(D)'.lower() or token.lower() in 'Poor'.lower():
                        confs['D'] += math.exp(logprob)
                        break

            conf = 1.0 * confs['A']/4 + (0.1) * confs['B']/4 + (-1.0) * confs['C']/4 + (-2.0) * confs['D']/4
            return conf
            
        all_reward = self_eval(gt_ans=gt_ans, response=action, question_input=pre_state.all_prompt)
        return all_reward
    
def perform_action(state, action, stop=None):
    new_all_prompt = f"{state.all_prompt} {action} "
    last_response = action
    return State(new_all_prompt, last_response, stop)

def simulate(node, max_simulation_steps=2):
    current_node = node
    generated = ""
    steps = 0
    while steps < max_simulation_steps and not current_node.is_terminal():
        if current_node.children:
            current_node = max(current_node.children, key=lambda c: c.value)
            generated += " " + current_node.state.last_response
        else:
            # 临时生成动作（不加入树）
            actions = get_actions(current_node.state, break_down=False, args=current_node.args)
            if actions:
                action_text, stop, _ = actions[0]
                new_state = perform_action(current_node.state, action_text, stop)
                generated += " " + action_text
                # 不创建新节点，仅用于模拟
            else:
                break
        steps += 1
    return generated, current_node

def mcts(initial_state, gt_ans=None, args=None):
    root = Node(initial_state, gt_ans=gt_ans, args=args)
    # print("Starting MCTS ...")

    for _iter in range(args.mcts_iters):
        # print(f"Iteration of MCTS {_iter} ...")
        node = root
        depth = 0
        while node.children is not None and len(node.children) > args.num_actions and depth < args.depth:
            node = node.best_child(args.exploration_weight)
            depth += 1

        if len(node.children) > args.num_actions or node.is_terminal() or node.depth >= args.depth: #end judge
            continue
        node.expand()
        generated, node = simulate(node, max_simulation_steps = args.max_simulation_steps) # 传递到的子节点
        reward = get_reward(node.state, generated, gt_ans=node.gt_ans, reward_model=args.actor_model, reward_model_name=args.model_name)
        node.backpropagate(reward)

    return root, root.best_child().state.last_response

def get_perference_data(root=None, args=None):
    def get_preference_pairs(node):
        if len(node.children) < 2:
            return None
        # print("Collecting preference pairs.")
        max_child = max(node.children, key=lambda x: x.value)
        min_child = min(node.children, key=lambda x: x.value)
        prefix_text = node.state.all_prompt
        accept = max_child.state.last_response
        reject = min_child.state.last_response
        if max_child.value - min_child.value < args.value_threshold or calculate_overlap_ratio(accept, reject) > args.overlap_limit or max_child.value < args.accept_value:
#             print("No qualified preference pair found.", {calculate_overlap_ratio(accept, reject)}, min_child.value, max_child.value, max_child.prob)
            return None
        accept_chat = extract_rounds(prefix_text, accept)
        reject_chat = extract_rounds(prefix_text, reject)
        try:
            assert len(accept_chat) == len(reject_chat) and len(accept_chat) != 0, f'接收和拒绝样本的长度不一致{prefix_text}, \n, {accept},\n, {reject}, \n, {accept_chat},\n {reject_chat}'
        except Exception as e:
            print(e)
            return None
        return {"conversations":accept_chat[:-1], "chosen":accept_chat[-1], "rejected": reject_chat[-1] }

    def extract_rounds(prefix_text, response): # chat_tamplate_prompt->chat prompt
        post_prompt = prefix_text + response
        post_prompt +=  '' if '<|eot_id|>' in post_prompt else '<|eot_id|>'
        # 使用正则表达式匹配每一轮对话,llama中新的模型样式
        pattern = r'<\|start_header_id\|\>(user|assistant|system|tool|ipython)<\|end_header_id\|\>(.*?)(?=<\|eot_id\|\>|$)'
        matches = re.findall(pattern, post_prompt, re.DOTALL)
        rounds = []
        for match in matches:
            role = match[0].strip()
            if role == 'ipython': role = 'tool'
            content = match[1].strip()
            # clear special_tokens
            special_tokens_pattern = r'<\|eot_id\|>|<\|end_of_text\|>|<\|eom_id\|>'
            content = re.sub(special_tokens_pattern, '', content)
            if role == 'system':  # remove cutting knowledge date
                if content.startswith('Cutting Knowledge Date'):
                    content = content[62:].strip()
            rounds.append({"role": role, "content": content})
        return rounds

    nodes_to_visit = [root]
    preference_pairs = []  # 用于存储所有的 preference_pair

    if args.save_preference_dir is not None:
        # 遍历所有节点并收集 preference_pair
        while nodes_to_visit:
            current_node = nodes_to_visit.pop()
            preference_pair = get_preference_pairs(current_node)
            if preference_pair:
                preference_pairs.append(preference_pair)  # 将 preference_pair 添加到列表中
            nodes_to_visit.extend(current_node.children)
        
    return preference_pairs

def print_tree(node, indent=""):
    """
    Recursively prints the tree starting from the given node.
    """
    # Print the current node's information
    print(f"{indent}Depth: {node.depth}, Visits: {node.visits}, Value: {node.value}, Prob: {node.prob} Response: {node.state.last_response}")

    # Recursively print the children
    for child in node.children:
        print_tree(child, indent + "  ")

def process_item(item, args=None):
    posted_question, gt_ans = item
    initial_state = State(all_prompt=posted_question)
    tree_root, best_action = mcts(initial_state, gt_ans=gt_ans, args=args)
#     print_tree(tree_root)
    perference_data = get_perference_data(tree_root, args=args)
    return perference_data

def get_dpo_data_by_mcts(args=None, dataset_path=None):
    # init
    print('get starting step 3 mcts \n\n')
    vllm_server = VLLMServer(model_path = args.model_path, num_gpus=args.num_gpus)
    try:
        vllm_server.start()
        args.actor_model, args.model_name, base_tokenizer = vllm_server.get_client()
        tool_dataset, _ = init_dataset(dataset_path=dataset_path, tokenizer=base_tokenizer)

        all_preference_pairs = []
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(process_item, item, args) for item in tqdm(tool_dataset)]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing MCST"):
                try:
                    result = future.result()
                    all_preference_pairs.extend(result)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    
        preference_dir = args.save_preference_dir.replace('.json',f'_{args.iter}.json')
        random.shuffle(all_preference_pairs) #shuffle 偏好数据 
        
        print('this is the save_preference_dir: ',preference_dir)
        print('this the len of preference data: ', len(all_preference_pairs) )
        with open(preference_dir, 'w', encoding='utf-8') as file:
            json.dump(all_preference_pairs, file, ensure_ascii=False, indent=2)  # 将整个列表写入文件
            file.write('\n')  # 可选：确保文件末尾换行
            
    except Exception as e:
        print(f'Error in MCTS: {str(e)}')
        raise e
    finally:
        print('Starting to stop the server...')
        vllm_server.stop()
        print('Server stopped. Proceeding to step 4: DPO training.\n\n')
         
if __name__ == "__main__":
    pass
