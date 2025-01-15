###用于生成prompt
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
input_prompt_token_limit = 2000

def input_prompt_1_func_total(dialogue_history_method, cen_decen_framework, area_resource_total, pos_update_prompt, response_total_list):
    user_prompt_1 = f'''
You are a central planner instructing two agents in a grid area to move to a specified area to collect resources. The entire map is a 1x1 coordinate point area with a number of known mission sites, namely,{{{{' "Vezzani"' ' "Teghime"' ' "Saint Cyprien"' ' "Corse ?"' ' "Salario (Ajaccio)"' ' "Corse"' ' "Asco"' ' "Casamaccioli"' ' "Pioggiola"' ' "Letia- Corse"' ' "Ortolo"' ' "Aullene"' ' "Letia - Corse"' ' "Tolla"' ' "Piccovaggia"' ' "Oso"' ' "Corte"' ' "Bonassa"' ' "Vignale"'}}}}, each mission location has several missions that need to be completed, all of which are considered successful, and each mission has its own task difficulty - easy or hard.
The two agents are marked as Agent0 and Agent1 respectively, and the highest difficulty of tasks that can be handled perfectly is simple and difficult respectively. Therefore, when Agent0 handles difficult tasks, the task completion rate will decrease, while Agent1 can handle any task perfectly, and the mobile operation is marked as: REACH ("Vezzani"), which means that the agent will reach the Vezzani region, and when the agent moves to a location, it will process the first five tasks of the current location, and then proceed to the next task according to the dispatch instruction. Moving between two locations takes time and task processing takes no time.

Label the number of tasks remaining at each task location and their corresponding difficulty in order as: Location Info: {{{{ "Vezzani": 2, hard, hard;  "Teghime": 1, easy; .....}}}}.
The location and status information of each agent is represented as {{"Agent0": "Tolla" (idle)", "Agent1": "Aullene" (moving)"}}. If the agent is not idle, no task is assigned.

Your task is to instruct three or two agents to move to the designated location to process the tasks so that all tasks are completed as quickly as possible and with a high completion rate. At the end of each turn, the agent provides an information update for the next sequence of actions. Your job is to get your agents to cooperate better.

Learn from the previous steps. Don't just repeat, but understand why the state changed or let the cycle continue. Avoid getting stuck in a cycle of action.

Therefore, the current information for each location is: {area_resource_total}
Each agent location is: {pos_update_prompt}

Your response should include only the specified action plan and not be in any other format: {{{{Agent0 : reach (xxx), Agent1 : reach (xxx)}}}}, if Agent1 or Agent0 is not idle {{{{Agent0 : reach (xxx), Agent1 : reach (None)}}}} indicates that Agent1 is not idle and does not need to be assigned. The planned area of arrival should be included in the possible destination of the drone.
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

######prompt更改成自己的动作模式
def input_prompt_local_agent_HMAS2_dialogue_func(area_resource_total,pos_update_prompt_local_agent, pos_update_prompt_other_agent,
                                                 central_response, response_total_list,
                                                 dialogue_history_list, dialogue_history_method):

    user_prompt_1 = f'''
You are a drone agent in a multi-agent system, stationed on a square area of 1x1 represented by coordinates. There are several task locations distributed among them, and each location has several tasks to be processed. When you go to a location, you process up to the first five tasks (no time consumed), but it takes a certain amount of time to get from one location to another, and when the number of tasks remaining in a location is 0, there are no tasks to be processed in that location. The ability of each drone to handle the task is different, resulting in different quality of completion.
The central planner coordinates all agents to achieve the goal: to complete the entire map task as quickly and as high-quality as possible.
Label the number of tasks remaining at each task location and their corresponding difficulty in order as: Location Info: {{{{ "Vezzani": 2, hard, hard;  "Teghime": 1, easy; .....}}}}.
Therefore, the current information for each location is: {area_resource_total}
Your current status is {{pos_update_prompt_local_agent}}.
The current status of all other agents is: {{pos_update_prompt_other_agent}}.
The state and action pairs on each step are:
Learn from the previous steps. Instead of simply repeating the action, understand why the state changes or stays in an endless loop. Avoid getting stuck in a cycle of action.
The current action plan of the central planner is: {{central_response}}.
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
Label the number of tasks remaining at each task location and their corresponding difficulty in order as: Location Info: {{{{ "Vezzani": 2, hard, hard;  "Teghime": 1, easy; .....}}}}.
Therefore, the current information for each location is: {area_resource_total}
Your current status is {{pos_update_prompt_local_agent}}.
The current status of all other agents is: {{pos_update_prompt_other_agent}}.
The state and action pairs on each step are: {{state_action_prompt}}
Learn from the previous steps. Instead of simply repeating the action, understand why the state changes or stays in an endless loop. Avoid getting stuck in a cycle of action.
The current action plan of the central planner is: {{central_response}}.
Please evaluate the given plan. If you agree, answer "I agree" without any additional words. If not, briefly explain your opposition to central planning.Your response:
   '''
    return user_prompt_1

######prompt更改成自己的动作模式
def input_prompt_local_agent_DMAS_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent,
                                                dialogue_history, response_total_list,
                                                dialogue_history_list,dialogue_history_method):

    user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.
  All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_green, square[0.5, 0.5])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, square[1.5, 1.5])"}}

  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
  '''
    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method == '_wo_any_dialogue_history':
        pass
    elif dialogue_history_method in (
    '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
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
                state_action_prompt_next = f'State{i + 1}:\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
                if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        user_prompt_1 = f'''
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects in your square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets and boxes.
    All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target.
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


    [Action Output Instruction]
    Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move..."}}.
    Include an agent only if it has a task next.
    Example#1: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_green, square[0.5, 0.5])"}}

    Example#2: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, square[1.5, 1.5])"}}

    The previous dialogue history is: {{{dialogue_history}}}
    Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
    Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
    Propose exactly one action for yourself at the **current** round.
    End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan, must strictly follow [Action Output Instruction]!
    Your response:
    '''

    return user_prompt_1

def LLM_summarize_func(state_action_prompt_next_initial, model_name):
  prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
  messages = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt1}]
  response = GPT_response(messages, model_name)
  return response




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

