# from env1_create import *
from env import *
from llm import *
from user_prompt_generation import *
# from env import EnvWrapper, TOP_LEFT, TOP_RIGHT, BOTTOM_CENTER, BOTTOM_RIGHT, BOTTOM_LEFT, TOP_CENTER, region_center, \
#     region_dict
import matplotlib.pyplot as plt
import json
# import numpy as np
from env_new import MultiDroneTaskEnv

# 预计添加的一些架构
# cen_decen_framework = 'DMAS', 'HMAS-1', 'CMAS', 'HMAS-2'
# dialogue_history_method = '_w_all_dialogue_history', '_wo_any_dialogue_history', '_w_only_state_action_history'
dialogue_history_method ='_w_only_state_action_history'
cen_decen_framework = 'HMAS-2'
# dialogue_history_method ='_wo_any_dialogue_history'
# cen_decen_framework = 'CMAS'



def plan_from_response(initial_response):
    response_dict = json.loads(initial_response)
    plans = [globals()[response_dict[key].split('(')[1].split(')')[0]] for key in response_dict]

    return plans


def pos_update_func_local_agent( uav_pos, idx, env):

    pos_dict = {}
    next_pos_area = {0: [], 1: [], 2: []}
    pos_update_prompt_other_agent = ''
    pos_update_prompt_local_agent = ''

    for idx, pos in enumerate(uav_pos):
        region = env.find_region(pos)  # 获取当前位置的区域标识符

        # 使用字典模拟 switch 语句，并将字符串全大写
        region_name = {
            10: "TOP_LEFT",
            11: "TOP_CENTER",
            12: "TOP_RIGHT",
            13: "CENTER_LEFT",
            14: "CENTER",
            15: "CENTER_RIGHT",
            16: "BOTTOM_LEFT",
            17: "BOTTOM_CENTER",
            18: "BOTTOM_RIGHT"
        }.get(region, "UNKNOWN")  # 默认值为 "UNKNOWN"
        pos_dict[idx] = region_name

    for i in range(3):
        valid_plan = []
        cur_pos = uav_pos[i]
        next_pos_list = []  # 用于存储当前 UAV 可以到达的区域名称


        for j in range(9):
            next_position = cur_pos + np.array(env.uav_move_map[j]) * env.max_move_step
            region = env.find_region(next_position)
            if region not in valid_plan:
                valid_plan.append(region)
        next_pos_area[i] = valid_plan

        for item in next_pos_area[i]:
            region_name = {
                10: "TOP_LEFT",
                11: "TOP_CENTER",
                12: "TOP_RIGHT",
                13: "CENTER_LEFT",
                14: "CENTER",
                15: "CENTER_RIGHT",
                16: "BOTTOM_LEFT",
                17: "BOTTOM_CENTER",
                18: "BOTTOM_RIGHT"
            }.get(item, "UNKNOWN")  # 默认值为 "UNKNOWN"
            next_pos_list.append(region_name)  # 将区域名称追加到列表中

        next_pos_str = '"' + '","'.join(next_pos_list) + '"'  # 将列表中的区域名称转换为字符串
        if i == idx:
            pos_update_prompt_local_agent += f"""{{"Agent{i}":"{pos_dict[i]}", "in the next iteration, I can reach areas as follows: [{next_pos_str}]"}}"""
        else:
            pos_update_prompt_other_agent += f"""{{"Agent{i}":"{pos_dict[i]}", "in the next iteration, I can reach areas as follows: [{next_pos_str}]"}}"""

    return pos_update_prompt_local_agent, pos_update_prompt_other_agent
#用于每回合与大模型对话
def conversation(user_prompt_list, response_total_list, token_num_count_list, dialogue_history_list, poi_map_info, uav_pos, env):

    
    

    pos_update_prompt = pos_to_text(uav_pos, env)
    plans = []
    print("A new aonversation start !!!!!##############################################################")

    # pos_update_prompt = {"Agent0": "CENTER",
    #                        "Agent1": "CENTER",
    #                        "Agent2": "CENTER"}
    # 第一轮的无人车位置信息可以直接输入，后面需要根据pos_to_text去更新
    if cen_decen_framework in ('CMAS', 'HMAS-2'):
        user_prompt_1 = input_prompt_1_func_total(dialogue_history_method, cen_decen_framework, area_resource_total,
                                                  pos_update_prompt, response_total_list)
        user_prompt_list.append(user_prompt_1)
        # if cen_decen_framework == 'CMAS':
        #     messages = message_construct_func(user_prompt_list, response_total_list, dialogue_history_method)
        # else:
        #     messages = message_construct_func([user_prompt_1], [],dialogue_history_method)
        messages = message_construct_func([user_prompt_1], [], '_w_all_dialogue_history')
        # 接收回复
        response, token_num_count = GPT_response(messages,'gpt-4')  # 'gpt-4' or 'gpt-3.5-turbo-0301' or 'gpt-4-32k' or 'gpt-3' or 'gpt-4-0613'
        print('Initial response: ', response)
        plans = plan_from_response(response)
        ###check_valid
        valid = False
        while valid == False:
            for idx in range(3):
                valid, _ = env.check_valid(idx,plans[idx])

                if valid == False:
                    print('{} --------------------错误无人机序号------------------------'.format(idx))
                    feedback = 'This is the feedback from your previous plans.{} is not valid.Your previous response is invalid because the unexpected plan area for Agent{} does not exist in uav{}\'s possible destination list, please correct the problem!Your answer should only include specified action plan without any other word in the following format: {{"Agent0":"reach(xxx)","Agent1":"reach(xxx)","Agent2":"reach(xxx)"}}.And the plan area reached should be included in the uav\'s possible destinations.Now give your corrected plan:'.format(response, idx, idx)
                    messages = message_construct_func([user_prompt_1,feedback], [response], '_w_all_dialogue_history')
                    response, token_num_count = GPT_response(messages,
                                                         'gpt-4')  # 'gpt-4' or 'gpt-3.5-turbo-0301' or 'gpt-4-32k' or 'gpt-3' or 'gpt-4-0613'
                    print('Modified response: ', response)
                    plans = plan_from_response(response)
                    break
        token_num_count_list.append(token_num_count)


        if cen_decen_framework == 'HMAS-2':
            print('--------HMAS-2 method starts--------')
            dialogue_history = f'Central Planner: {response}\n'
            prompt_list_dir = {}
            response_list_dir = {}
            local_agent_response_list_dir = {}
            local_agent_response_list_dir['feedback1'] = ''

            for idx in range(3):
                prompt_list_dir[f'Agent[{idx}'] = []
                response_list_dir[f'Agent[{idx}'] = []
                pos_update_prompt_local_agent, pos_update_prompt_other_agent = pos_update_func_local_agent( uav_pos, idx, env)
                local_reprompt = input_prompt_local_agent_HMAS2_dialogue_func(
                    pos_update_prompt_local_agent, pos_update_prompt_other_agent, response,
                    response_total_list, dialogue_history_list, dialogue_history_method)
                prompt_list_dir[f'Agent[{idx}'].append(local_reprompt)
                messages = message_construct_func( prompt_list_dir[f'Agent[{idx}'], response_list_dir[f'Agent[{idx}'], '_w_all_dialogue_history')
                response_local_agent, token_num_count = GPT_response(messages, 'gpt-4')
                token_num_count_list.append(token_num_count)
                if response_local_agent != 'I Agree':
                    local_agent_response_list_dir['feedback1'] += f'Agent[{idx}]: {response_local_agent}\n'  # collect the response from all the local agents
                    dialogue_history += f'Agent[{idx}]: {response_local_agent}\n'
            ###对机器人的反馈进行检查
            if local_agent_response_list_dir['feedback1'] != '':
                #这里提示需要修改一下 改一下格式就行
                local_agent_response_list_dir[
                    'feedback1'] += '\nThis is the feedback from local agents. If you find some errors in your previous plan, try to modify it. Otherwise, output the same plan as before. The output should have the same json format {"Agent0":"reach(xxx)","Agent1":"reach(xxx)","Agent2":"reach(xxx)"}, as above. Do not explain, just directly output Your response:'
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
                pos_update_prompt_local_agent, pos_update_prompt_other_agent = pos_update_func_local_agent( uav_pos, idx, env)

                user_prompt_1 = input_prompt_local_agent_DMAS_dialogue_func(pos_update_prompt_local_agent,
                                                                            pos_update_prompt_other_agent,
                                                                            dialogue_history,
                                                                            response_total_list,
                                                                            pg_state_list,
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

def main():
    save_img = True
    # 使用新的环境类，并传入所需的参数
    csv_path = './fire_data.csv'
    env = MultiDroneTaskEnv(csv_path, num_drones=2, photos_per_mission=5)
    uav_num = env.num_drones


    uav0_collects = []
    uav1_collects = []
    total = []
    # 输出结果
    # print("Total sum of poi_value:", total_poi_value)  # 如果不再需要，可以移除
    for episode in range(5):
        state = env.reset()

        uav_pos = state  # 更新为从 reset 返回的状态
        # info = env.get_map_info()  # 如果不再需要，可以移除
        time_step = 0
        # record(episode, time_step, poi_pos, poi_value, uav_pos, uav_num, sensing_range)  # 如果不再需要，可以移除

        ###############
        done = False
        dialogue_history_list = []
        user_prompt_list = []  # The record list of all the input prompts
        response_total_list = []  # The record list of all the responses
        token_num_count_list = []  # The record list of the length of token
        plans = []
        while time_step < 9 and not done:
            time_step += 1

            ############### 这里接入大模型
            if time_step % 3 == 1:
                plans = conversation(user_prompt_list, response_total_list, token_num_count_list, dialogue_history_list, None, uav_pos, env)
            print(plans)
            ########
            for uav_id in range(uav_num):
                plan = plans[uav_id]
            

            state, reward, done, info = env.step(plans)

            # poi_value = env.poi_value  # 如果不再需要，可以移除
            uav_pos = state
           
        print('uav0:{}, uav1:{}, total: {}'.format(env.acc_uav_total, env.acc_uav_total, env.acc_uav_total))



if __name__ == '__main__':
    main()
"""
HAMS-2
CMAS
三次PLAN，每次3步 与random（合法区域随机选）比较剩余资源量
观察每个agent的资源收集量
"""
