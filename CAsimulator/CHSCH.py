import numpy as np
import random

# 定义路网矩阵
matrix = np.array([
    [2, 3, 5, 2, 3, 5, 2, 3],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [1, 1, 4, 1, 1, 4, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [1, 1, 4, 1, 1, 4, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [3, 2, 5, 3, 2, 5, 3, 2]
])

# 获取入口、出口和中途目的地的位置
entrances = np.argwhere(matrix == 2)
exits = np.argwhere(matrix == 3)
destinations = []

# 目的地是数字4周围的元胞
for position in np.argwhere(matrix == 4):
    # 获取上下左右的元胞
    x, y = position
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    destinations.extend(neighbors)

# 选择随机入口、目的地和出口
random_entrance = tuple(random.choice(entrances))
random_destination = tuple(random.choice(destinations))
random_exit = tuple(random.choice(exits))

# 输出选择的点
(random_entrance, random_destination, random_exit)
