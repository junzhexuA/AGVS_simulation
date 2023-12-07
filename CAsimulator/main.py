import numpy as np
import pandas as pd
import random
from Task import *
from AGV import *
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

def load_map(filepath):
    matrix = np.array(pd.read_excel(filepath, header=None))
    entrances = np.argwhere(matrix == 2)
    exits = np.argwhere(matrix == 3)
    destinations = np.argwhere(matrix == 4)
    return matrix, entrances, exits, destinations

def generate_task(matrix, entrances, destinations, arrival_time_list):

    # 生成任务列表
    task_list = []
    for i in range(len(arrival_time_list)):
        task_list.append(task(tuple(random.choice(entrances)),tuple(random.choice(destinations)),arrival_time_list[i],matrix))
        # 若任务时间和起点和之前的任务相同，则重置该任务
        if i>0:
            while task_list[i].entrance == task_list[i-1].entrance and task_list[i].arrival_time == task_list[i-1].arrival_time:
                task_list[i] = task(tuple(random.choice(entrances)),tuple(random.choice(destinations)),arrival_time_list[i],matrix) #逻辑待修改

    # 输出任务列表
    for i in range(len(task_list)):
        print(f"任务{i}开始时间:",task_list[i].arrival_time, f"任务{i}起点：", task_list[i].entrance, f"任务{i}投递口：", task_list[i].destination, f"任务{i}出口：", task_list[i].exit)

    return task_list

def simulate(frame, ax, agv_matrix, task_list, arrival_time_list):
    global agvs, tmp

    print("第", frame, "步")
    agv_matrix, agvs = simulate_step(agv_matrix, agvs)

    # Output AGV information
    for j in range(len(agvs)):
        print(f"AGV{j}当前位置：", agvs[j].position, f"AGV{j}当前速度：", agvs[j].speed, f"AGV{j}路径：", agvs[j].path)

    to_remove = []
    for j in tmp:
        if j == frame:
            print(f"第{j}时间步有新任务到达")
            agvs.append(AGV(task_list[arrival_time_list.index(j)].full_path[0], 0, False, task_list[arrival_time_list.index(j)].full_path, 1, direction=None, 
                            entrance=task_list[arrival_time_list.index(j)].entrance, destination=task_list[arrival_time_list.index(j)].stop_position, 
                            exit=task_list[arrival_time_list.index(j)].exit, arrival_time=task_list[arrival_time_list.index(j)].arrival_time))
            to_remove.append(j)  # 将要删除的时间步添加到临时列表中
            for k in range(len(agvs)):
                if len(agvs[k].path) > 0:
                    if agvs[k].state == 1:
                        agv_matrix[agvs[k].position[0]][agvs[k].position[1]] = -1
                    if agvs[k].state == 0:
                        agv_matrix[agvs[k].position[0]][agvs[k].position[1]] = -2
                # Output AGV information
                print(f"AGV{k}当前位置：", agvs[k].position, f"AGV{k}当前速度：", agvs[k].speed, f"AGV{k}起点：", 
                      agvs[k].entrance, f"AGV{k}投递位置：", agvs[k].destination, f"AGV{k}出口：", agvs[k].exit, 
                      f"AGV{k}任务开始时间步：", agvs[k].arrival_time, f"AGV{k}状态：", agvs[k].state)
                
    # 在循环结束后从原始列表中删除要删除的时间步
    for j in to_remove:
        tmp.remove(j)
    print(tmp)
    print(agv_matrix)


    # 仅移除 -1 和 -2 对应的图形元素
    for artist in ax.findobj(match=plt.Circle):
        artist.remove()

    # 绘制 -1 和 -2 对应的位置
    for i in range(len(agv_matrix)):
        for j in range(len(agv_matrix[i])):
            if agv_matrix[i][j] == -1:
                circle = plt.Circle((j, i), 0.45, color='red')
                ax.add_patch(circle)
            elif agv_matrix[i][j] == -2:
                circle = plt.Circle((j, i), 0.45, color='blue')
                ax.add_patch(circle)

    return [ax]

def main(matrix, entrances, exits, destinations, arrival_time_list, task_list, agvs, V_max, time_Step):
    agv_matrix = copy.deepcopy(matrix)
    cmap = colors.ListedColormap(['blue','red','white','white','white','white','black','gray'])
    bounds = [-2,-1, 0, 1, 2, 3, 4, 5, 6]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 初始化地图画布
    fig, ax = plt.subplots(figsize=(5, 5))

    # 绘制初始 agv_matrix，排除 -1 和 -2 的部分
    for i in range(len(agv_matrix)):
        for j in range(len(agv_matrix[i])):
            if agv_matrix[i][j] not in [-1, -2]:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, color=cmap(norm(agv_matrix[i][j])))
                ax.add_patch(rect)

    # 重新设置坐标轴的限制和比例，包括边框
    ax.set_xlim(-0.5, len(agv_matrix[0])-0.5)
    ax.set_ylim(len(agv_matrix)-0.5, -0.5)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xticks(np.arange(-0.5, len(agv_matrix[0])-0.5, 1))
    ax.set_yticks(np.arange(-0.5, len(agv_matrix)-0.5, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_title('AGV Simulation')

    # 进行仿真，每次刷新间隔1000毫秒
    animation = FuncAnimation(fig, simulate, fargs=(ax, agv_matrix, task_list, arrival_time_list), frames=time_Step, repeat=False, blit=False, interval=1000)
    plt.show()

    # 保存动画为GIF
    #animation.save('agv_simulation_big.gif', writer='imagemagick', fps=10)


if __name__ == '__main__':
    random.seed(42)
    # 路网矩阵 
    matrix, entrances, exits, destinations = load_map('CAsimulator\map_file\small_map.xlsx')

    # 生成随机到达任务时间列表
    arrival_time_list = [random.randint(0,10) for i in range(10)]
    arrival_time_list.sort()

    # 生成任务列表
    task_list = generate_task(matrix, entrances, destinations, arrival_time_list)

    # 初始化AGV列表,v_max为最大速度, time_step为仿真时间步长
    agvs = []
    V_max = 1
    time_Step = 200
    tmp = copy.deepcopy(arrival_time_list) #拷贝到达时间列表
    
    main(matrix, entrances, exits, destinations, arrival_time_list, task_list, agvs, V_max, time_Step)






