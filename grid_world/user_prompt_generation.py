###用于生成prompt
import tiktoken
import pandas as pd
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
input_prompt_token_limit = 2000
csv_path = './democsv.csv'
def generate_location_content(csv_path):
    # 从CSV文件中读取地点名和坐标
    df = pd.read_csv(csv_path)
    locations = df['location'].unique()
    
    # 生成地点内容，包含坐标信息
    location_content_list = []
    for location in locations:
        location_data = df[df['location'] == location].iloc[0]
        x, y = location_data['x'], location_data['y']
        location_content_list.append(f"{location} (x: {x}, y: {y})")
    
    location_content = ", ".join(location_content_list)
    return f"Known disaster sites include: {location_content}."

def input_prompt_1_func_total(dialogue_history_method, cen_decen_framework, area_resource_total, pos_update_prompt, response_total_list, num_drones):
    # 动态生成 UAV 标识
    uav_identifiers = ', '.join([f'UAV{i}' for i in range(num_drones)])
    
    # 动态生成地点内容
    location_content = generate_location_content(csv_path)
    
    user_prompt_1 = f'''
You are a coordinator in a distributed system responsible for managing {num_drones} UAVs responding to disaster sites. Your task is to efficiently guide them to specific locations where disasters have occurred. The map is a 1x1 coordinate grid with several known disaster sites, including: {location_content}. Each site requires immediate attention, and tasks must be completed within 30 minutes of the disaster occurring.

These UAVs are identified as {uav_identifiers}. The operation is denoted as: REACH ("Vezzani"), indicating that the UAV will move to the Vezzani region. Upon arrival, the UAV will perform necessary tasks to mitigate the disaster. Note that moving between locations takes time, and task completion must be efficient to meet the 30-minute deadline.

You should label the number of tasks remaining at each location and their urgency as follows: Location Info: {{{{ "Vezzani": 2, urgent; "Teghime": 1, urgent; .....}}}}.
The status and location of each UAV are represented as: {{"UAV0": "Tolla" (idle)", "UAV1": "Aullene" (moving)", ...}}. If a UAV is not idle, no new task is assigned.

Your objective is to instruct these {num_drones} UAVs to move to designated locations to complete tasks as efficiently and effectively as possible. At the end of each turn, UAVs will provide an update for the next sequence of actions. Your role is to ensure optimal cooperation among the UAVs to maximize efficiency and meet the 30-minute deadline.

Learn from previous steps. Avoid mere repetition; understand why states change or cycles continue. Prevent getting stuck in action loops.

Therefore, the current information for each location is: {area_resource_total}
Each UAV's location is: {pos_update_prompt}

Your answer should include only the specified action plan, not any other format, and no quotation marks: {{{{UAV0 : reach (xxx), UAV1 : reach (xxx), ...}}}}. If any UAV is not idle, use {{{{UAV0 : reach (xxx), UAV1 : reach (None), ...}}}} to indicate that the UAV is not idle and does not need to be assigned. The planned area of arrival should be included in the possible destinations for the UAV.
Now, plan your next step:
    '''
    token_num_count = len(enc.encode(user_prompt_1))
    if dialogue_history_method == '_wo_any_dialogue_history' or cen_decen_framework == 'CMAS':
        pass
    elif dialogue_history_method in ('_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
        if dialogue_history_method == '_w_only_state_action_history' and cen_decen_framework != 'CMAS':
            state_action_prompt = ''
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = f'plan{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
                if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        # elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
        #     pass
        elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
            pass

    token_num_count = len(enc.encode(user_prompt_1))
    print(token_num_count)
    return user_prompt_1


def input_prompt_local_agent_HMAS2_dialogue_func(area_resource_total,pos_update_prompt_local_agent, pos_update_prompt_other_agent,
                                                 central_response, response_total_list,
                                                 dialogue_history_list, dialogue_history_method):

    user_prompt_1 = f'''
You are a drone agent in a multi-agent system, stationed on a square area of 1x1 represented by coordinates. There are several task locations distributed among them, and each location has several tasks to be processed. When you go to a location, you process up to the first five tasks (no time consumed), but it takes a certain amount of time to get from one location to another, and when the number of tasks remaining in a location is 0, there are no tasks to be processed in that location. The ability of each drone to handle the task is different, resulting in different quality of completion.
The central planner coordinates all agents to achieve the goal: to complete the entire map task as quickly and as high-quality as possible.
Label the number of tasks remaining at each task location and their time remaining in order as: Location Info: {{{{ "Vezzani": 2, 27, 28;  "Teghime": 1, 27; .....}}}}.This means that Vezzani has two tasks left, each corresponding to 27, 28 minutes of remaining time.
Therefore, the current information for each location is: {area_resource_total}
Your current status is {pos_update_prompt_local_agent}.
The current status of all other agents is: {pos_update_prompt_other_agent}.
The state and action pairs on each step are:
Learn from the previous steps. Instead of simply repeating the action, understand why the state changes or stays in an endless loop. Avoid getting stuck in a cycle of action.
The current action plan of the central planner is: {central_response}.
Please evaluate the given plan. If you agree, answer "I agree" without any additional words. If not, briefly explain your opposition to central planning.Your response:
   '''

    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method == '_wo_any_dialogue_history':
        pass
    elif dialogue_history_method in (
            '_w_only_state_action_history', '_w_all_dialogue_history'):
        if dialogue_history_method == '_w_only_state_action_history':
            state_action_prompt = ''
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = f'State{i + 1}: Plan{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
                if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        # elif dialogue_history_method == '_w_compressed_dialogue_history':
        #     state_action_prompt = ''
        #     for i in range(len(response_total_list) - 1, -1, -1):
        #         dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        #         state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        #         # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        #         if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
        #             state_action_prompt = state_action_prompt_next
        #         else:
        #             break
        elif dialogue_history_method == '_w_all_dialogue_history':
            state_action_prompt = ''
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = f'State{i + 1}: Dialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n' + state_action_prompt
                if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        user_prompt_1 = f'''
You are a drone agent in a multi-agent system, stationed on a square area of 1x1 represented by coordinates. There are several task locations distributed among them, and each location has several tasks to be processed. When you go to a location, you process up to the first five tasks (no time consumed), but it takes a certain amount of time to get from one location to another, and when the number of tasks remaining in a location is 0, there are no tasks to be processed in that location. The ability of each drone to handle the task is different, resulting in different quality of completion.
The central planner coordinates all agents to achieve the goal: to complete the entire map task as quickly and as high-quality as possible.
Label the number of tasks remaining at each task location and their time remaining in order as: Location Info: {{{{ "Vezzani": 2, 27, 28;  "Teghime": 1, 27; .....}}}}.This means that Vezzani has two tasks left, each corresponding to 27, 28 minutes of remaining time.
Therefore, the current information for each location is: {area_resource_total}
Your current status is {pos_update_prompt_local_agent}.
The current status of all other agents is: {pos_update_prompt_other_agent}.
The state and action pairs on each step are: {state_action_prompt}
Learn from the previous steps. Instead of simply repeating the action, understand why the state changes or stays in an endless loop. Avoid getting stuck in a cycle of action.
The current action plan of the central planner is: {central_response}.
Please evaluate the given plan. If you agree, answer "I agree" without any additional words. If not, briefly explain your opposition to central planning.Your response:
   '''
    return user_prompt_1


# def LLM_summarize_func(state_action_prompt_next_initial, model_name):
#   prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
#   messages = [{"role": "system", "content": "You are a helpful assistant."},
#               {"role": "user", "content": prompt1}]
#   response = GPT_response(messages, model_name)
#   return response




def message_construct_func(user_prompt_list, response_total_list, dialogue_history_method):
    messages = []
    if f'{dialogue_history_method}' == '_w_all_dialogue_history':
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        # print('length of user_prompt_list', len(user_prompt_list))
        for i in range(len(user_prompt_list)):
            messages.append({"role": "user", "content": user_prompt_list[i]})
            if i < len(user_prompt_list) - 1:
                messages.append({"role": "assistant", "content": response_total_list[i]})
        # print('Length of messages', len(messages))
    elif f'{dialogue_history_method}' in ('_wo_any_dialogue_history', '_w_only_state_action_history'):
        # 这种情况下 role：system 这部分内容是恒定的
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": user_prompt_list[-1]})
        # print('Length of messages', len(messages))
    return messages

