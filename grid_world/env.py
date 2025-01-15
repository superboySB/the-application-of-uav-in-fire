import math
import copy
import numpy as np

HIGH = 0
MEDIUM = 1
LOW = 2

MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_UP = 2
MOVE_DOWN = 3
MOVE_LEFT_UP = 4
MOVE_LEFT_DOWN = 5
MOVE_RIGHT_UP = 6
MOVE_RIGHT_DOWN = 7
STAY_STILL = 8

TOP_LEFT = 10
TOP_CENTER = 11
TOP_RIGHT = 12
CENTER_LEFT = 13
CENTER = 14
CENTER_RIGHT = 15
BOTTOM_LEFT = 16
BOTTOM_CENTER = 17
BOTTOM_RIGHT = 18

region_dict = {
    TOP_LEFT: "top_left",
    TOP_CENTER: "top_center",
    TOP_RIGHT: "top_right",
    CENTER_LEFT: "center_left",
    CENTER: "center",
    CENTER_RIGHT: "center_right",
    BOTTOM_LEFT: "bottom_left",
    BOTTOM_CENTER: "bottom_center",
    BOTTOM_RIGHT: "bottom_right",
}
region_center = {
    BOTTOM_LEFT: [1.0 / 6, 1.0 / 6],
    BOTTOM_CENTER: [3.0 / 6, 1.0 / 6],
    BOTTOM_RIGHT: [5.0 / 6, 1.0 / 6],
    CENTER_LEFT: [1.0 / 6, 3.0 / 6],
    CENTER: [3.0 / 6, 3.0 / 6],
    CENTER_RIGHT: [5.0 / 6, 3.0 / 6],
    TOP_LEFT: [1.0 / 6, 5.0 / 6],
    TOP_CENTER: [3.0 / 6, 5.0 / 6],
    TOP_RIGHT: [5.0 / 6, 5.0 / 6],
}

data_level = {
    TOP_LEFT: HIGH, TOP_CENTER: LOW, TOP_RIGHT: HIGH,
    CENTER_LEFT: LOW, CENTER: MEDIUM, CENTER_RIGHT: LOW,
    BOTTOM_LEFT: HIGH, BOTTOM_CENTER: LOW, BOTTOM_RIGHT: HIGH
}

data_amount = {
    LOW: [[0, 0.2], 5],
    MEDIUM: [[0.4, 0.6], 10],
    HIGH: [[0.8, 1.0], 20]
}

plan_space = [
    TOP_LEFT, TOP_CENTER, TOP_RIGHT, CENTER_LEFT, CENTER, CENTER_RIGHT, BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT
]



class EnvWrapper(object):
    def __init__(self, ):
        data = np.load("poi_info.npz")
        poi_pos_and_value_arr = np.array(data['poi_pos_and_value'])
        self.poi_pos = poi_pos_and_value_arr[:, 0:2]
        self.poi_init_value = poi_pos_and_value_arr[:, 2]
        self.poi_num = self.poi_pos.shape[0]

        self.poi_value = copy.deepcopy(self.poi_init_value)

        self.uav_num = 3
        self.uav_init_pos = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        self.uav_pos = copy.deepcopy(self.uav_init_pos)
        self.uav_sensing_range = 0.235
        self.uav_move_len = 0.07
        # 收集资源量
        self.uav0_collect = 0
        self.uav1_collect = 0
        self.uav2_collect = 0

        self.max_move_step = 3
        self.interval_step = 5
        self.uav_move_map = {
            MOVE_LEFT: [-self.uav_move_len, 0],
            MOVE_RIGHT: [self.uav_move_len, 0],
            MOVE_UP: [0, self.uav_move_len],
            MOVE_DOWN: [0, -self.uav_move_len],
            MOVE_LEFT_UP: [-self.uav_move_len / math.sqrt(2), self.uav_move_len / math.sqrt(2)],
            MOVE_LEFT_DOWN: [-self.uav_move_len / math.sqrt(2), -self.uav_move_len / math.sqrt(2)],
            MOVE_RIGHT_UP: [self.uav_move_len / math.sqrt(2), self.uav_move_len / math.sqrt(2)],
            MOVE_RIGHT_DOWN: [self.uav_move_len / math.sqrt(2), -self.uav_move_len / math.sqrt(2)],
            STAY_STILL: [0, 0]
        }



    def reset(self):
        self.uav_pos = np.array(self.uav_init_pos)
        self.poi_value = copy.deepcopy(self.poi_init_value)
        self.uav0_collect = 0
        self.uav1_collect = 0
        self.uav2_collect = 0
        info = self.get_map_info()
        state = {
            'uav_pos': self.uav_pos,
            'region_poi': info
        }

        return state

    def get_map_info(self):
        # data_level = {
        #     0: HIGH, 1: LOW, 2: HIGH,
        #     3: LOW, 4: MEDIUM, 5: LOW,
        #     6: HIGH, 7: LOW, 8: HIGH
        # }
        #
        # data_amount = {
        #     LOW: [[0, 0.2], 5],
        #     MEDIUM: [[0.4, 0.6], 10],
        #     HIGH: [[0.8, 1.0], 20]
        # }
        pre_poi_num = 0
        poi_map_info = {key: 0 for key in data_level.keys()}
        for region in data_level.keys():
            poi_num = data_amount[data_level[region]][1]
            poi_value_sum = np.sum(self.poi_value[pre_poi_num:(pre_poi_num + poi_num)])
            poi_map_info[region] = poi_value_sum
            pre_poi_num += poi_num
        return poi_map_info

    def collect_data(self, pos):
        total_data = 0
        for i in range(self.poi_num):
            poi_pos = self.poi_pos[i]
            if np.linalg.norm(pos - poi_pos) <= self.uav_sensing_range:
                total_data += min(self.poi_value[i], 0.1)
                self.poi_value[i] = max(0, self.poi_value[i] - 0.1)
        return total_data

    def find_region(self, pos):
        assert len(pos) == 2
        x, y = pos
        if y < 1.0 / 3:
            if x < 1.0 / 3:
                return BOTTOM_LEFT
            elif x < 2.0 / 3:
                return BOTTOM_CENTER
            else:
                return BOTTOM_RIGHT
        elif y < 2.0 / 3:
            if x < 1.0 / 3:
                return CENTER_LEFT
            elif x < 2.0 / 3:
                return CENTER
            else:
                return CENTER_RIGHT
        else:
            if x < 1.0 / 3:
                return TOP_LEFT
            elif x < 2.0 / 3:
                return TOP_CENTER
            else:
                return TOP_RIGHT

    def check_valid(self, uav_id, plan):
        valid_plan = []
        cur_pos = self.uav_pos[uav_id]
        for i in range(9):
            next_pos = cur_pos + np.array(self.uav_move_map[i]) * self.max_move_step
            region = self.find_region(next_pos)
            if region not in valid_plan:
                valid_plan.append(region)
        return plan in valid_plan, valid_plan

    def convert_action(self, plan, pos):
        assert len(pos) == 2
        assert plan in region_center.keys()
        region = self.find_region(pos)
        if region == plan:
            while True:
                action = np.random.choice(list(self.uav_move_map.keys()))
                next_pos = pos + self.uav_move_map[action]
                next_x = next_pos[0]
                next_y = next_pos[1]
                if next_y > 0 and next_y < 1 and next_x > 0 and next_x < 1:
                    next_region = self.find_region(next_pos)
                    if next_region == region:
                        # still stay in goal region
                        return action
        else:
            x, y = pos
            region_x, region_y = region_center[plan]
            eps = 0.01
            if abs(y - region_y) < eps:
                if abs(x - region_x) < eps:
                    action = STAY_STILL
                elif x < region_x:
                    action = MOVE_RIGHT
                else:
                    action = MOVE_LEFT
            elif y < region_y:
                if abs(x - region_x) < eps:
                    action = MOVE_UP
                elif x < region_x:
                    action = MOVE_RIGHT_UP
                else:
                    action = MOVE_LEFT_UP
            else:
                if abs(x - region_x) < eps:
                    action = MOVE_DOWN
                elif x < region_x:
                    action = MOVE_RIGHT_DOWN
                else:
                    action = MOVE_LEFT_DOWN
            next_pos = pos + self.uav_move_map[action]
            next_x = next_pos[0]
            next_y = next_pos[1]
            return action

    def update_uav_pos(self, uav_pos):
        self.uav_pos = np.array(uav_pos)

    def step(self, plans):
        new_plans = []
        for uav_id in range(self.uav_num):
            plan = plans[uav_id]
            flag, valid_plan = self.check_valid(uav_id, plan)
            if not flag:
                plan = np.random.choice(valid_plan)
            new_plans.append(plan)

        uav_collect_data = []
        for _ in range(self.interval_step):
            uav_pos = []
            for uav_id in range(self.uav_num):
                cur_uav_pos = self.uav_pos[uav_id]
                action = self.convert_action(new_plans[uav_id], cur_uav_pos)
                move_dir = self.uav_move_map[action]
                next_pos = [cur_uav_pos[0] + move_dir[0], cur_uav_pos[1] + move_dir[1]]

                next_x = next_pos[0]
                next_y = next_pos[1]

                uav_pos.append(next_pos)
                total_data = self.collect_data(next_pos)
                if uav_id == 0:
                    self.uav0_collect += total_data
                if uav_id == 1:
                    self.uav1_collect += total_data
                if uav_id == 2:
                    self.uav2_collect += total_data
                uav_collect_data.append(total_data)

            self.update_uav_pos(uav_pos)

        reward = np.mean(uav_collect_data)
        info = self.get_map_info()
        done = np.sum(self.poi_value) < 0.01
        state = {
            'uav_pos': self.uav_pos,
            'region_poi': info
        }

        return state, reward, done, info
