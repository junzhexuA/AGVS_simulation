import numpy as np
import pandas as pd

class GridWorldEnv:
    def __init__(self, excel_file):
        # 从 Excel 文件中读取地图
        self.grid = pd.read_excel(excel_file, header=None).to_numpy()
        self.action_space = ['up', 'down', 'left', 'right', 'stay']
        self.reset()

    def reset(self):
        # 如果没有指定起始位置和目标位置，则随机选择
        if start_position is None:
            start_position = self.choose_random_start()

        if target_position is None:
            target_position = self.choose_random_target()

        self.current_position = start_position
        self.target_position = target_position

        return self.get_state()

    def choose_random_start(self):
        # 从入口位置中随机选择一个
        start_positions = np.argwhere(self.grid == 2)  # 入口用数字 2 表示
        return start_positions[np.random.choice(len(start_positions))]

    def choose_random_target(self):
        # 从目标位置中随机选择一个
        target_positions = np.argwhere(self.grid == 4)  # 目标点用数字 4 表示
        return target_positions[np.random.choice(len(target_positions))]
    
    def step(self, action):
        # 根据动作更新状态
        # 返回新状态、奖励和是否结束
        # ...
        return new_state, reward, done

    def get_state(self):
        # 获取当前状态的张量表示
        # ...
        return state_tensor

    # 其他
    def calculate_new_position(self, action):
        # 根据当前位置和动作计算新位置
        # 确保新位置在地图内且不是障碍物
        # ...
        return new_position

    def calculate_reward(self, position):
        # 根据新位置计算奖励
        # ...
        return reward

    def is_done(self, position):
        # 检查是否达到结束条件，如到达目标点或碰到障碍物
        # ...
        return done

    def render(self):
        # 可选：实现一个方法来可视化环境状态
        # ...
