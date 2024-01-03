import numpy as np
import pandas as pd
import random
from Task import *
from AGV import *
from greedy_solution import *
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

def load_map_data(filepath):
    matrix = np.array(pd.read_excel(filepath, header=None))
    entrances = np.argwhere(matrix == 2)
    exits = np.argwhere(matrix == 3)
    destinations = np.argwhere(matrix == 4)
    return matrix, entrances, exits, destinations

def generate_task(matrix, entrances, destinations, arrival_time_list, mode='random'):
    '''
    生成任务列表
    '''
    global sku_info, entrances_dic, deliverPos_dic
    ST_Table = np.expand_dims(map, axis=0)
    if mode == 'random':
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

    if mode == 'instance':
        # 生成任务列表
        task_list = []
        for i in range(len(arrival_time_list)):
            task_list.append(task(tuple(entrances_dic[sku_info[i][2]]),tuple(deliverPos_dic[solution[i]]),arrival_time_list[i],matrix))
            # task_list.append(task(tuple(entrances_dic[sku_info[i][2]] + (arrival_time_list[i],)),tuple(deliverPos_dic[solution[i]]),arrival_time_list[i],matrix,ST_Table))
            # 若任务时间和起点和之前的任务相同，则重置该任务
            if i>0:
                while task_list[i].entrance == task_list[i-1].entrance and task_list[i].arrival_time == task_list[i-1].arrival_time:
                    task_list[i] = task(tuple(entrances_dic[sku_info[i][2]]),tuple(deliverPos_dic[solution[i]]),arrival_time_list[i]+1,matrix)
                    # task_list[i] = task(tuple(entrances_dic[sku_info[i][2]]+ (arrival_time_list[i],)),tuple(deliverPos_dic[solution[i]]),arrival_time_list[i]+1,matrix,ST_Table)
        # 输出任务列表
        for i in range(len(task_list)):
            print(f"任务{i}开始时间:",task_list[i].arrival_time, f"任务{i}起点：", task_list[i].entrance, f"任务{i}投递口：", task_list[i].destination, f"任务{i}出口：", task_list[i].exit)
    return task_list

def simulate(frame, ax, agv_matrix, task_list, arrival_time_list):
    global agvs, tmp, space_time_table
        
    print("第", frame, "步")
    if frame > 0:
        agv_matrix, agvs = simulate_step(agv_matrix, agvs)
    to_remove = []
    flag = 0
    count = 0
    for j in tmp:
        if j == frame:
            count += 1
            index = [i for i, a in enumerate(arrival_time_list) if a == j] # 修改
            agvs.append(AGV(task_list[index[flag]].full_path[0], 0, False, task_list[index[flag]].full_path, 1, direction=None, 
                        entrance=task_list[index[flag]].entrance, destination=task_list[index[flag]].stop_position, 
                        exit=task_list[index[flag]].exit, arrival_time=task_list[index[flag]].arrival_time))
            flag += 1
            to_remove.append(j)  # 将要删除的时间步添加到临时列表中
    if count > 0:        
        print(f"第{frame}时间步有{count}个任务到达")
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
        print(f"AGV{k}当前位置：", agvs[k].position, f"AGV{k}当前速度：", agvs[k].speed, f"AGV{k}路径：", agvs[k].path)     
    # 在循环结束后从原始列表中删除要删除的时间步
    for j in to_remove:
        tmp.remove(j)
    print(agv_matrix)
    space_time_table.append(agv_matrix)


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
    animation = FuncAnimation(fig, simulate, fargs=(ax, agv_matrix, task_list, arrival_time_list), frames=time_Step, repeat=False, blit=False, interval=200)
    plt.show()

    # 保存动画为GIF
    # animation.save('agv_simulation_big.gif', writer='imagemagick', fps=10)

# 任务日志记录任务信息和每步AGV信息
def log_task():
    pass

if __name__ == '__main__':
    #random.seed(42)

    # 读取文件
    file_path = "CAsimulator\instance\s_2.xlsx"
    map_path = "CAsimulator\map_file\small_map.xlsx"
    sku_data = load_data(file_path,sheet_name="sku_info")
    sku_info = np.array(sku_data.iloc[:,1:])
    demand = np.array(load_data(file_path,sheet_name="demand"))
    map = load_map(map_path)
    entrancePos = [i for i in zip(np.where(map==2)[0],np.where(map==2)[1])]
    deliverPos = [i for i in zip(np.where(map==4)[0],np.where(map==4)[1])]
    entrances_dic = {i:entrancePos[i] for i in range(len(entrancePos))}
    deliverPos_dic = {i:deliverPos[i] for i in range(len(deliverPos))}
    ST_Table = np.expand_dims(map, axis=0)
    space_time_table = []

    # 贪婪算法求解
    solution, demand_new, ST_Table, total_time,path = greedy_search(map, sku_info,demand,entrances_dic,deliverPos_dic,ST_Table)
    '''total_time = calculate_time(solution,sku_info,entrances_dic,deliverPos_dic,map)
    print('总用时：',total_time)'''
    print(solution)
    print(demand_new)
    print('总用时：',total_time)
    print(ST_Table.shape)
    print("-"*150+"\n")
    
    # 用txt文件记录结果
    with open('CAsimulator\instance\s_1_solution.txt', 'w') as f:
        f.write(f"结果：{solution}\n")
        f.write("-"*150+"\n")
        f.write(f"总用时：{total_time}\n")
        f.write("-"*150+"\n")
        f.write(f"路径：{path}\n")
        f.write("-"*150+"\n")
        f.write(f"ST表：{ST_Table.shape}\n")

    # 路网矩阵 
    matrix, entrances, exits, destinations = load_map_data('CAsimulator\map_file\small_map.xlsx')

    '''# 生成随机到达任务时间列表
    arrival_time_list = [random.randint(0,10) for i in range(200)]
    arrival_time_list.sort()

    # 生成任务列表
    task_list = generate_task(matrix, entrances, destinations, arrival_time_list)'''
    
    # 读取任务到达时间
    arrival_time_list = [i for i in sku_info[:,1]]
    arrival_time_list.sort()

    # 生成任务列表
    task_list = generate_task(matrix, entrances, destinations, arrival_time_list, mode='instance')

    # 初始化AGV列表,v_max为最大速度, time_step为仿真时间步长
    agvs = []
    V_max = 1
    time_Step = 250
    tmp = copy.deepcopy(arrival_time_list) #拷贝到达时间列表
    
    main(matrix, entrances, exits, destinations, arrival_time_list, task_list, agvs, V_max, time_Step)
    space_time_table = np.array(space_time_table[1:])
    print(space_time_table.shape)






