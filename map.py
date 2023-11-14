import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
# 决策单元 1
# 取货单元 2
# 出口单元 3
# 投递单元 4
# 预留单元 5
# 其他单元 0


class Map:
    def __init__(self):
        excel_path='map.xlsx'
        df=pd.read_excel(excel_path,header=None)
        self.map=np.array(df)
    def rand_start(self):
        start=np.where(self.map==2)
        index=random.choice(range(len(start[0])))
        return (start[0][index],start[1][index])
    def rand_end(self):
        end=np.where(self.map==4)
        index=random.choice(range(len(end[0])))
        return (end[0][index],end[1][index])
        

def visualize_map(map, start, end, path):
    # 创建一个新的图像
    plt.figure()
    # 创建自定义的颜色映射
    cmap = colors.ListedColormap(['white', 'yellow', 'black','gray','lightyellow'])

    # 创建归一化对象
    bounds = [-0.5,0.5, 3.5, 4.5, 5.5,6.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 使用自定义的颜色映射和归一化对象来显示图像
    plt.imshow(map, cmap=cmap, norm=norm)
    # 设置x轴和y轴的刻度，每隔1画一个网格
    plt.xticks(np.arange(-.5, map.shape[1], 1), [])
    plt.yticks(np.arange(-.5, map.shape[0], 1), [])
    # 添加网格
    plt.grid(color='black', linewidth=1)

    # 绘制起点和终点
    plt.scatter([start[1], end[1]], [start[0], end[0]], c='red')

    # 绘制路径
    if path is not None:
        path = np.array(path)
        path_2d = np.array([p[:2] for p in path])
        plt.plot([start[1], path_2d[0,1]], [start[0], path_2d[0,0]], c='blue')
        plt.plot(path_2d[:, 1], path_2d[:, 0], c='blue')

    # 显示图像
    plt.show()

    # 分别提取行号、列号和时刻
    rows, cols, times = zip(*path[:-1])

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制路径
    ax.plot(rows, cols, times, marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('Row')
    ax.set_ylabel('Column')
    ax.set_zlabel('Time')

    # 显示图形
    plt.show()

def threeD_path(path):
    # 创建3D图形
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

   # 用于存储所有点以检测重合
    all_points = set()
    collision_points = []

    # 遍历每条路径并绘制
    for single_path in path:
        # 分别提取行号、列号和时刻
        rows, cols, times = zip(*single_path[:-1])

        # 检测重合点
        for point in single_path[:-1]:
            if point in all_points:
                collision_points.append(point)
            all_points.add(point)
        # 绘制路径
        ax.plot(rows, cols, times, marker='o')
    

    # 标记重合点
    for point in collision_points:
        ax.scatter(*point, color='red', marker='x')
        ax.text(point[0], point[1], point[2], 'collision', color='red')

    # 设置坐标轴标签
    ax.set_xlabel('Row')
    ax.set_ylabel('Column')
    ax.set_zlabel('Time')

    # 显示图形
    plt.show()
    print("冲突点：",collision_points)