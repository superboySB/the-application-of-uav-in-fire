from env import *
from llm import *
from user_prompt_generation import *
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
# import numpy as np

def generate_valid_area(uav_id, env):
    valid_plan = []
    cur_pos = env.uav_pos[uav_id]
    for i in range(9):
        next_pos = cur_pos + np.array(env.uav_move_map[i]) * env.max_move_step
        region = env.find_region(next_pos)
        if region not in valid_plan:
            valid_plan.append(region)
    return valid_plan


def plot_uav_collection(uav0, uav1, uav2, filename='uav_random_collection.png'):
    labels = ['UAV 0', 'UAV 1', 'UAV 2', 'Total']
    values = [uav0, uav1, uav2, uav0 + uav1 + uav2]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['#FF6347', '#FF8C00', '#FFD700', '#FF4500'])  # 使用暖色系颜色
    plt.xlabel('UAVs and Total')
    plt.ylabel('Collected Value')
    plt.title('UAV Collection Summary')

    for i, value in enumerate(values):
        plt.text(i, value + 0.1, str(value), ha='center')

    # 保存图片到代码目录
    plt.savefig(filename, bbox_inches='tight')

    # 如果不需要显示图片，可以注释掉下面这行
    #plt.show()


def main():
    save_img = True
    env = EnvWrapper()
    uav_num = env.uav_num
    sensing_range = env.uav_sensing_range
    poi_pos = env.poi_pos

    uav0_collects = []
    uav1_collects = []
    uav2_collects = []
    total = []
    total_poi_value = np.sum(env.poi_value)
    # 输出结果
    print("Total sum of poi_value:", total_poi_value)
    for episode in range(5):
        state = env.reset()
        poi_value = env.poi_value
        uav_pos = env.uav_pos
        info = env.get_map_info()
        time_step = 0

        #record(episode, time_step, poi_pos, poi_value, uav_pos, uav_num, sensing_range)
        ###############
        done = False
        plans = [CENTER,CENTER,CENTER]
        while time_step < 9 and not done:
            time_step += 1
            if time_step % 3 == 1:
                for uav_id in range(uav_num):
                    valid_plan = generate_valid_area(uav_id, env)
                    plan = np.random.choice(valid_plan)
                    plans[uav_id] = plan

                print('Plans:', plans)


            state, reward, done, info = env.step(plans)

            poi_value = env.poi_value
            uav_pos = env.uav_pos
            #record(episode, time_step, poi_pos, poi_value, uav_pos, uav_num, sensing_range)
        x = env.uav0_collect
        uav0_collects.append(x/total_poi_value)
        uav1_collects.append(env.uav1_collect/total_poi_value)
        uav2_collects.append(env.uav2_collect/total_poi_value)
        total.append((env.uav0_collect + env.uav1_collect + env.uav2_collect) / total_poi_value)
            #plot_uav_collection(env.uav0_collect, env.uav1_collect, env.uav2_collect, filename='uav_random_collection.png')
        # 计算每个 UAV 的平均收集值
        print('uav0:{}, uav1:{}, uav2:{}, total: {}'.format(env.uav0_collect, env.uav1_collect, env.uav2_collect,
                                                            env.uav0_collect + env.uav1_collect + env.uav2_collect))

    uav0_avg = np.mean(uav0_collects)
    uav1_avg = np.mean(uav1_collects)
    uav2_avg = np.mean(uav2_collects)
    uav0_var = np.var(uav0_collects)
    uav1_var = np.var(uav1_collects)
    uav2_var = np.var(uav2_collects)

    # 计算总的平均值和方差
    total_avg = np.mean(total)
    total_var = np.var(total)
    print(total_avg, total_var)

    # 绘制条形图的标签、平均值和方差
    labels = ['UAV 0', 'UAV 1', 'UAV 2', 'Total']
    averages = [uav0_avg, uav1_avg, uav2_avg, total_avg]
    variances = [uav0_var, uav1_var, uav2_var, total_var]
    # 如果方差计算后小于0，则将其设置为0
    variances = np.maximum(variances, 0)
    # 绘制图表
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, averages, yerr=np.sqrt(variances), capsize=10,
                   color=['#FF6347', '#FF8C00', '#FFD700', '#FF4500'])
    # 在条形图上显示具体数值
    for bar, avg in zip(bars, averages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{avg:.2f}', ha='center', va='bottom')

    plt.xlabel('UAV')
    plt.ylabel('Average Collection')
    plt.title('Average UAV Collection with Variance over 5 Episodes')
    plt.savefig('uav_random_collection.png', bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    main()
"""
HAMS-2
三次PLAN，每次3步 与random（合法区域随机选）比较剩余资源量
"""