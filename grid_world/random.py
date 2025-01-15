import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio

class MultiDroneTaskEnv(gym.Env):
    def __init__(self, csv_path, num_drones=8, photos_per_mission=5):
        super(MultiDroneTaskEnv, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.df.sort_values(by=['year', 'month'], inplace=True) 
        self.locations = self.df['lieu'].unique()
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_drones, 2), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(self.locations)] * num_drones)
        self.num_drones = num_drones
        self.photos_per_mission = photos_per_mission  # 每次任务拍摄的照片数量
        self.drone_positions = np.random.rand(num_drones, 2)
        
        # 使用CSV文件中的difficulty列
        if 'difficulty' not in self.df.columns:
            raise ValueError("The 'difficulty' column is missing in the CSV file.")
        
        self.task_queue = self.df.groupby('lieu').apply(lambda x: x).reset_index(drop=True)
        self.initial_task_count = len(self.task_queue)  # 初始的任务总数
        
        # 定义每个无人机能处理的最高难度
        self.drone_capabilities = ['easy'] * 4 + ['difficult'] * 4
        
        # 从CSV文件中读取每个地点的坐标
        self.location_coords = {row['lieu']: np.array([row['x'], row['y']]) for _, row in self.df.iterrows()}
        
        # 初始化无人机状态
        self.drone_states = ['idle'] * num_drones  # 状态有： 'idle', 'moving', 'tasking'
        self.drone_targets = [None] * num_drones  # 记录每个无人机的目标地点
        
        # 初始化acc相关变量
        self.acc_uav_total = 0
        self.acc_task_total = self.calculate_high_acc_task()  # 预先计算所有任务的acc_task_sum
        self.acc_uav_count = 0  # 用于计算acc_uav的均值
        
        # 初始化无人机派遣次数
        self.drone_dispatch_count = np.zeros(num_drones, dtype=int)
        
        # 初始化 task_acc 总和
        self.task_acc_total = 0
        
    def calculate_total_acc_task(self):
        acc_task_sum = 0
        for _, task in self.df.iterrows():
            if task['difficulty'] == 'easy':
                acc_task_sum += task['LowClass2Acc']
            else:
                acc_task_sum += task['Class2AccLight']
        return acc_task_sum
    
    def calculate_high_acc_task(self):
        acc_task_sum = 0
        for _, task in self.df.iterrows():
                acc_task_sum += task['Class2AccLight']
        return acc_task_sum
        
    def reset(self):
        self.drone_positions = np.random.rand(self.num_drones, 2)
        self.task_queue = self.df.groupby('lieu').apply(lambda x: x).reset_index(drop=True)
        self.drone_states = ['idle'] * self.num_drones
        self.drone_targets = [None] * self.num_drones
        self.acc_uav_total = 0
        self.acc_uav_count = 0
        self.drone_dispatch_count = np.zeros(self.num_drones, dtype=int)
        self.task_acc_total = 0
        return self.drone_positions
    
    def update_task_queue(self, target_location, drone_index):
        location_tasks = self.task_queue[self.task_queue['lieu'] == target_location]
        completed_tasks = 0
        acc_uav_sum = 0
        
        for _, task in location_tasks.iterrows():
            task_difficulty = task['difficulty']
            if self.drone_capabilities[drone_index] == 'easy':
                uav_iou = task['LowClass2Acc']
            else:
                uav_iou = task['Class2AccLight']
            
            # if task_difficulty == 'easy':
            #     task_iou = task['LowClass2Acc']
            # else:
            #     task_iou = task['Class2AccLight']
            
            # if self.drone_capabilities[drone_index] == task_difficulty:
            #     acc_uav = task_iou
            # else:
            #     acc_uav = min(uav_iou, task_iou)
            
            acc_uav_sum += uav_iou
            self.acc_uav_count += 1
            
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
                # 仅从有任务的地点中选择目标位置
                available_locations = [loc for loc in self.locations if len(self.task_queue[self.task_queue['lieu'] == loc]) > 0]
                
                if available_locations:
                    target_location = np.random.choice(available_locations)
                    self.drone_targets[i] = target_location
                    self.drone_states[i] = 'moving'
                    self.drone_dispatch_count[i] += 1  # 增加派遣次数
                else:
                    print(f"No available tasks for Drone {i}.")
            
            if self.drone_states[i] == 'moving':
                self.move_drone(i)
            
            if self.drone_states[i] == 'tasking':
                completed_tasks, acc_uav_sum = self.update_task_queue(self.drone_targets[i], i)
                rewards[i] = completed_tasks
                self.acc_uav_total += acc_uav_sum
                # 任务完成后，选择新的任务
                self.drone_states[i] = 'idle'
            
        done = len(self.task_queue) == 0
        metric1 = self.acc_uav_total / self.acc_task_total if self.acc_task_total > 0 else 0
        acc = self.acc_uav_total / self.acc_uav_count if self.acc_uav_count > 0 else 0
        return self.drone_positions, rewards, done, {'metric1': metric1, 'acc': acc}
    
    def render(self, mode='human'):
        print(f"Drone Positions: {self.drone_positions}")
        print(f"Drone States: {self.drone_states}")
        print(f"Remaining Tasks: {len(self.task_queue)}")
        print(f"Initial Task Count: {self.initial_task_count}")
        print(f"Drone Dispatch Count: {self.drone_dispatch_count}")
        print(f"Total Dispatches: {np.sum(self.drone_dispatch_count)}")
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

    def plot_initial_distribution(self):
        plt.figure(figsize=(6, 6))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Initial Distribution")
        
        # 绘制任务地点
        for loc, coord in self.location_coords.items():
            remaining_tasks = len(self.task_queue[self.task_queue['lieu'] == loc])
            plt.scatter(*coord, c='red', marker='x')
            plt.text(coord[0], coord[1], f'{loc} ({remaining_tasks})', fontsize=8, ha='right')
        
        # 绘制无人机位置
        for i, pos in enumerate(self.drone_positions):
            plt.scatter(*pos, c='blue', marker='o')
            plt.text(pos[0], pos[1], f'Drone {i}', fontsize=8, ha='right')
        
        plt.savefig('initial_distribution.png')
        plt.show()

# 使用环境并生成GIF
csv_path = 'fire_data.csv'
df = pd.read_csv('fire_data.csv')
print(df.columns)
env = MultiDroneTaskEnv(csv_path, photos_per_mission=5)  # 设置每次任务拍摄的照片数量
state = env.reset()
done = False
frames = []

# 绘制初始分布图
env.plot_initial_distribution()

frame_number = 0
max_steps = 50  # 设置最大轮数

#while not done and frame_number < max_steps:
while not done:    
    actions = env.action_space.sample()  # 随机选择动作
    state, rewards, done, info = env.step(actions)
    env.render()
    env.plot(frame_number)
    frames.append(f'frame_{frame_number}.png')
    frame_number += 1

# 生成GIF
with imageio.get_writer('drone_simulation.gif', mode='I', duration=0.5) as writer:
    for filename in frames:
        image = imageio.imread(filename)
        writer.append_data(image)

# 清理临时文件
import os
for filename in frames:
    os.remove(filename)

# 输出最终的指标
print(frame_number)
final_metric1 = info['metric1']
final_acc = info['acc']
final_acc_std = env.calculate_high_acc_task() / env.acc_uav_count
print(f"Final metric1: {final_metric1}")
print(f"Final acc: {final_acc}")
print(f"Final acc_std: {final_acc_std}")


# 运行代码 ok
# 差值越大任务越难，差值越小任务越简单  对任务添加属性 ok
# 任务的属性（难度， 地点） ok
# 拍摄数量作为超参设置 ok
# metric固定轮次计算满足率 ，即iou（是否满足清晰度，不满足 (西格玛（iou_right-iou）/iou_right),满足则为1）的差值， 
