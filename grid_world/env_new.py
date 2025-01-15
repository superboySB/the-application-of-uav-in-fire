import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
# from llm import *
from user_prompt_generation import *
# # from env import EnvWrapper, TOP_LEFT, TOP_RIGHT, BOTTOM_CENTER, BOTTOM_RIGHT, BOTTOM_LEFT, TOP_CENTER, region_center, \
# #     region_dict
# import matplotlib.pyplot as plt
# import json
# # import numpy as np
import re

# cen_decen_framework = 'DMAS', 'HMAS-1', 'CMAS', 'HMAS-2'
# dialogue_history_method = '_w_all_dialogue_history', '_wo_any_dialogue_history', '_w_only_state_action_history'
dialogue_history_method ='_w_only_state_action_history'
cen_decen_framework = 'HMAS-2'
# dialogue_history_method ='_wo_any_dialogue_history'
# cen_decen_framework = 'CMAS'

class MultiDroneTaskEnv(gym.Env):
    def __init__(self, csv_path, num_drones=2, photos_per_mission=5):
        super(MultiDroneTaskEnv, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.df.sort_values(by=['year', 'month'], inplace=True) 
        self.locations = self.df['lieu'].unique()
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_drones, 2), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(self.locations)] * num_drones)
        
        # 打印 action_space 的信息
        print(f"Action Space: {self.action_space}")
        
        self.num_drones = num_drones
        self.photos_per_mission = photos_per_mission  # 每次任务拍摄的照片数量
        self.drone_positions = np.random.rand(num_drones, 2)
        
        # 使用CSV文件中的difficulty列
        if 'difficulty' not in self.df.columns:
            raise ValueError("The 'difficulty' column is missing in the CSV file.")
        
        self.task_queue = self.df.groupby('lieu').apply(lambda x: x).reset_index(drop=True)
        self.initial_task_count = len(self.task_queue)  # 初始的任务总数
        
        # 定义每个无人机能处理的最高难度
        self.drone_capabilities = ['easy', 'hard']
        
        # 从CSV文件中读取每个地点的坐标
        self.location_coords = {row['lieu']: np.array([row['x'], row['y']]) for _, row in self.df.iterrows()}
        
        # 初始化无人机状态
        self.drone_states = ['idle'] * num_drones  # 状态有： 'idle', 'moving', 'tasking'
        self.drone_targets = [None] * num_drones  # 记录每个无人机的目标地点
        
        # 初始化acc相关变量
        self.acc_uav_total = 0
        self.acc_task_total = self.calculate_total_acc_task()  # 预先计算所有任务的acc_task_sum
        
    def calculate_total_acc_task(self):
        acc_task_sum = 0
        for _, task in self.df.iterrows():
            if task['difficulty'] == 'easy':
                acc_task_sum += task['MeanIoUHeavy']
            else:
                acc_task_sum += task['MeanIoULight']
        return acc_task_sum
        
    def reset(self):
        self.drone_positions = np.random.rand(self.num_drones, 2)
        self.task_queue = self.df.groupby('lieu').apply(lambda x: x).reset_index(drop=True)
        self.drone_states = ['idle'] * self.num_drones
        self.drone_targets = [None] * self.num_drones
        self.acc_uav_total = 0
        return self.drone_positions
    
    def update_task_queue(self, target_location, drone_index):
        location_tasks = self.task_queue[self.task_queue['lieu'] == target_location]
        completed_tasks = 0
        acc_uav_sum = 0
        
        for _, task in location_tasks.iterrows():
            task_difficulty = task['difficulty']
            if self.drone_capabilities[drone_index] == 'easy':
                uav_iou = task['MeanIoUHeavy']
            else:
                uav_iou = task['MeanIoULight']
            
            if task_difficulty == 'easy':
                task_iou = task['MeanIoUHeavy']
            else:
                task_iou = task['MeanIoULight']
            
            if self.drone_capabilities[drone_index] == task_difficulty:
                acc_uav = task_iou
            else:
                acc_uav = uav_iou
            
            acc_uav_sum += acc_uav
            
            completed_tasks += 1
            self.task_queue = self.task_queue.drop(task.name)
            
            # 打印任务完成情况
            print(f"Drone {drone_index} completed task at {target_location} with difficulty {task_difficulty}.")
            
            if completed_tasks >= self.photos_per_mission:  # 使用超参数限制任务处理数量
                break
        
        self.drone_states[drone_index] = 'idle'  # 任务完成
        return completed_tasks, acc_uav_sum
    
    def move_drone(self, drone_index):
        target_location = self.drone_targets[drone_index]
        if target_location is None:
            return
        
        target_pos = self.location_coords[target_location]
        current_pos = self.drone_positions[drone_index]
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.2:
            direction = direction / distance  # 归一化方向
            self.drone_positions[drone_index] += direction * 0.2
        else:
            self.drone_positions[drone_index] = target_pos  # 到达目标位置
            self.drone_states[drone_index] = 'tasking'  # 开始任务

    def step(self, actions):
        rewards = np.zeros(self.num_drones)
        
        for i in range(self.num_drones):
            if self.drone_states[i] == 'idle':
                target_location = self.locations[actions[i]]
                self.drone_targets[i] = target_location
                self.drone_states[i] = 'moving'
            
            if self.drone_states[i] == 'moving':
                self.move_drone(i)
            
            if self.drone_states[i] == 'tasking':
                completed_tasks, acc_uav_sum = self.update_task_queue(self.drone_targets[i], i)
                rewards[i] = completed_tasks
                self.acc_uav_total += acc_uav_sum
            
        done = len(self.task_queue) == 0
        acc = self.acc_uav_total / self.acc_task_total if self.acc_task_total > 0 else 0
        return self.drone_positions, rewards, done, {'acc': acc}
    
    def render(self, mode='human'):
        print(f"Drone Positions: {self.drone_positions}")
        print(f"Drone States: {self.drone_states}")
        print(f"Remaining Tasks: {len(self.task_queue)}")
        print(f"Initial Task Count: {self.initial_task_count}")
        print(self.task_queue[['lieu', 'difficulty']].head())

    def plot(self, frame_number):
        plt.figure(figsize=(6, 6))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f"Frame {frame_number}")
        
        # 绘制任务地点
        for loc, coord in self.location_coords.items():
            remaining_tasks = len(self.task_queue[self.task_queue['lieu'] == loc])
            plt.scatter(*coord, c='red', marker='x')
            plt.text(coord[0], coord[1], f'{loc} ({remaining_tasks})', fontsize=8, ha='right')
        
        # 绘制无人机位置
        for i, pos in enumerate(self.drone_positions):
            plt.scatter(*pos, c='blue', marker='o')
            plt.text(pos[0], pos[1], f'Drone {i}', fontsize=8, ha='right')
        
        plt.savefig(f'frame_{frame_number}.png')
        plt.close()

def location_to_text():
    location_info = {}
    
    # 遍历每个地点
    for loc in env.locations:
        # 获取该地点的剩余任务
        tasks_at_location = env.task_queue[env.task_queue['lieu'] == loc]
        remaining_task_count = len(tasks_at_location)
        
        # 获取任务难度信息
        task_difficulties = tasks_at_location['difficulty'].tolist()
        
        # 构建字符串信息
        location_info[loc] = f"{remaining_task_count}, " + ", ".join(task_difficulties)
    
    # 将信息格式化为字符串，并在首尾加上 { 和 }
    location_info_str = "; ".join([f"{loc}: {info}" for loc, info in location_info.items()])
    location_info_str = "{" + location_info_str + "}"
    
    return location_info_str

def pos_to_text():
    agent_positions = {}
    
    # 遍历每个无人机
    for i, target_location in enumerate(env.drone_targets):
        # 获取无人机的工作状态
        drone_state = env.drone_states[i]
        
        if target_location is not None:
            # 如果无人机有目标地点，则使用目标地点
            agent_positions[f"Agent{i}"] = f"{target_location} ({drone_state})"
        else:
            # 如果没有目标地点，使用当前所在地
            current_pos = env.drone_positions[i]
            # 找到最近的地点名
            closest_location = None
            min_distance = float('inf')
            
            for loc, coord in env.location_coords.items():
                distance = np.linalg.norm(current_pos - coord)
                if distance < min_distance:
                    min_distance = distance
                    closest_location = loc
            
            agent_positions[f"Agent{i}"] = f"{closest_location} ({drone_state})"
    
    # 将信息格式化为字符串
    pos_info_str = "{" + ", ".join([f'"{agent}": "{location}"' for agent, location in agent_positions.items()]) + "}"
    
    return pos_info_str

def pos_update_func_local_agent(idx):
    pos_update_prompt_local_agent = ''
    pos_update_prompt_other_agent = ''
    
    # 遍历每个无人机
    for i in range(2):  # 假设有两个无人机
        # 获取无人机的工作状态和能力
        drone_state = env.drone_states[i]
        drone_capability = env.drone_capabilities[i]
        
        # 获取目标地点或当前位置
        if env.drone_targets[i] is not None:
            location = env.drone_targets[i]
            action = f"going to {location}"
        else:
            # 如果没有目标地点，使用当前所在地
            current_pos = env.drone_positions[i]
            closest_location = None
            min_distance = float('inf')
            
            for loc, coord in env.location_coords.items():
                distance = np.linalg.norm(current_pos - coord)
                if distance < min_distance:
                    min_distance = distance
                    closest_location = loc
            
            location = closest_location
            action = f"staying at {location}"
        
        # 构建字符串信息并加上标识
        agent_label = f"Agent{i}:"
        if i == idx:
            pos_update_prompt_local_agent = f"{agent_label} I am {drone_state}, I can handle {drone_capability} tasks perfectly at most, I am {action}"
        else:
            pos_update_prompt_other_agent = f"{agent_label} I am {drone_state}, I can handle {drone_capability} tasks perfectly at most, I am {action}"
    
    return pos_update_prompt_local_agent, pos_update_prompt_other_agent

def plan_from_response(response):
    # 使用正则表达式提取括号中的地点
    pattern = r'Agent\d+\s*:\s*reach\s*\(([^)]+)\)'
    matches = re.findall(pattern, response)
    
    # 将提取的地点转换为计划
    plans = []
    for i, match in enumerate(matches):
        location_name = match.strip()
        print(location_name)
        
        # 如果 location_name 为 None，使用当前目标地点
        if location_name.lower() == 'none':
            location_name = env.drone_targets[i]
        
        # 将地点名转换为索引
        if location_name in env.locations:
            location_index = np.where(env.locations == location_name)[0][0]
        else:
            # 如果没有找到匹配的地点名，自动匹配最接近的地点名
            closest_location = min(env.locations, key=lambda loc: levenshtein_distance(location_name, loc))
            location_index = np.where(env.locations == closest_location)[0][0]
            print(f"Location '{location_name}' not found. Using closest match: '{closest_location}'")
        
        plans.append(location_index)
    
    return plans

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # 初始化距离矩阵
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def conversation(user_prompt_list, response_total_list, token_num_count_list, dialogue_history_list):

    location_info = location_to_text()
    print(f"Location Info: {location_info}")
    uav_update_prompt = pos_to_text()
    print(f"Position Update: {uav_update_prompt}")  # 打印无人机位置信息
    plans = []
    print("A new conversation start !!!!!##############################################################")


    if cen_decen_framework in ('CMAS', 'HMAS-2'):
        user_prompt_1 = input_prompt_1_func_total(dialogue_history_method, cen_decen_framework, location_info,
                                                  uav_update_prompt, response_total_list)
        user_prompt_list.append(user_prompt_1)
        # if cen_decen_framework == 'CMAS':
        #     messages = message_construct_func(user_prompt_list, response_total_list, dialogue_history_method)
        # else:
        #     messages = message_construct_func([user_prompt_1], [],dialogue_history_method)
        
        messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history')
        print(messages)
        # 接收回复 待后续修改
        #response, token_num_count = GPT_response(messages,'gpt-4')  # 'gpt-4' or 'gpt-3.5-turbo-0301' or 'gpt-4-32k' or 'gpt-3' or 'gpt-4-0613'
        response = ''' Agent0 : reach (Pioggiola), Agent1 : reach (Pioggiola) '''
        print('Initial response: ', response)
        plans = plan_from_response(response)
        ###check_valid
        #token_num_count_list.append(token_num_count)


        if cen_decen_framework == 'HMAS-2':
            print('--------HMAS-2 method starts--------')
            dialogue_history = f'Central Planner: {response}\n'
            prompt_list_dir = {}
            response_list_dir = {}
            local_agent_response_list_dir = {}
            local_agent_response_list_dir['feedback1'] = ''

            for idx in range(2):
                prompt_list_dir[f'Agent[{idx}'] = []
                response_list_dir[f'Agent[{idx}'] = []
                pos_update_prompt_local_agent, pos_update_prompt_other_agent = pos_update_func_local_agent(idx)
                local_reprompt = input_prompt_local_agent_HMAS2_dialogue_func(
                    pos_update_prompt_local_agent, pos_update_prompt_other_agent, response,
                    response_total_list, dialogue_history_list, dialogue_history_method)
                prompt_list_dir[f'Agent[{idx}'].append(local_reprompt)
                messages = message_construct_func( prompt_list_dir[f'Agent[{idx}'], response_list_dir[f'Agent[{idx}'], '_w_all_dialogue_history')
                print(messages)
                response_local_agent, token_num_count = GPT_response(messages, 'gpt-4')
                token_num_count_list.append(token_num_count)
                if response_local_agent != 'I Agree':
                    local_agent_response_list_dir['feedback1'] += f'Agent[{idx}]: {response_local_agent}\n'  # collect the response from all the local agents
                    dialogue_history += f'Agent[{idx}]: {response_local_agent}\n'
            ###对机器人的反馈进行检查
            if local_agent_response_list_dir['feedback1'] != '':
                #这里提示需要修改一下 改一下格式就行
                local_agent_response_list_dir[
                    'feedback1'] += '\nThis is the feedback from local agents. If you find some errors in your previous plan, try to modify it. Otherwise, output the same plan as before. The output should have the same json format {{{{Agent0 : reach (xxx), Agent1 : reach (xxx)}}}}, if Agent1 or Agent0 is not idle {{{{Agent0 : reach (xxx), Agent1 : reach (None)}}}} indicates that Agent1 is not idle and does not need to be assigned, as above. Do not explain, just directly output Your response:'
                messages = message_construct_func(
                    [user_prompt_list[-1], local_agent_response_list_dir['feedback1']], [response],
                    '_w_all_dialogue_history')  # message construction 第一个列表长度为2，没有问题
                response_central_again, token_num_count = GPT_response(messages, 'gpt-4')
                token_num_count_list.append(token_num_count)
                print(f'Modified plan response:\n {response}')
            else:
                print(f'Plan:\n {response}')
                pass
            dialogue_history_list.append(dialogue_history)
    else:
        print("DMAS START!--------------------------  先不用管这个")
        match = None
        count_round = 0
        dialogue_history = ''
        response = '{}'
        while not match and count_round <= 3:
            count_round += 1
            for idx in range(3):
                pos_update_prompt_local_agent, pos_update_prompt_other_agent = pos_update_func_local_agent(idx)

                user_prompt_1 = input_prompt_local_agent_DMAS_dialogue_func(location_info,pos_update_prompt_local_agent,
                                                                            pos_update_prompt_other_agent,
                                                                            dialogue_history,
                                                                            response_total_list,
                                                                            dialogue_history_list,
                                                                            dialogue_history_method)
                user_prompt_list.append(user_prompt_1)
                print(f'User prompt: {user_prompt_1}\n')
                messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history')
                response, token_num_count = GPT_response(messages,model_name)  # 'gpt-4' or 'gpt-3.5-turbo-0301' or 'gpt-4-32k' or 'gpt-3' or 'gpt-4-0613'
                token_num_count_list.append(token_num_count)
                dialogue_history += f'[Agent[{idx}]: {response}]\n\n'
                print(f'response: {response}')
                plans = plan_from_response(response)
        dialogue_history_list.append(dialogue_history)
#添加最终的决策
    response_total_list.append(response)


    return plans



# 使用环境并生成GIF
csv_path = './fire_data.csv'
env = MultiDroneTaskEnv(csv_path, photos_per_mission=5)  # 设置每次任务拍摄的照片数量
state = env.reset()
done = False
frames = []
dialogue_history_list = []
user_prompt_list = []  # The record list of all the input prompts
response_total_list = []  # The record list of all the responses
token_num_count_list = []  # The record list of the length of token
plans = []
frame_number = 0
max_steps = 2  # 设置最大轮数

while not done and frame_number < max_steps:
#while not done:
    plans = conversation(user_prompt_list, response_total_list, token_num_count_list, dialogue_history_list)
    actions = plans  # 将 plans 直接赋值给 actions
    state, rewards, done, info = env.step(actions)
    env.render()
    env.plot(frame_number)
    frames.append(f'frame_{frame_number}.png')
    frame_number += 1
    
    # 打印 location_to_text 的结果


# 生成GIF
with imageio.get_writer('drone_simulation.gif', mode='I', duration=0.5) as writer:
    for filename in frames:
        image = imageio.imread(filename)
        writer.append_data(image)

# 清理临时文件
import os
for filename in frames:
    os.remove(filename)

# 输出最终的acc
final_acc = info['acc']
print(f"Final acc: {final_acc}")


# 运行代码 ok
# 差值越大任务越难，差值越小任务越简单  对任务添加属性 ok
# 任务的属性（难度， 地点） ok
# 拍摄数量作为超参设置 ok
# metric固定轮次计算满足率 ，即iou（是否满足清晰度，不满足 (西格玛（iou_right-iou）/iou_right),满足则为1）的差值， 
