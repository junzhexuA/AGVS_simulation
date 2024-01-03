from map import *
from stA_star import StAstar
from stA_star import update_StTable
from GoodsArrival import *
import random
import numpy as np
from greedy_solution import *
import time

#random.seed(42)

if __name__ == '__main__':
    raw_map = Map()

    '''#生成到达商品序列
    arrival_sequences = simulate_arrival(6, [100, 150, 200], 0.25, 10)
    #转化成起点信息
    start = covert2start(arrival_sequences)'''
    ''''start = [(9, 4, 0), (9, 7, 0), (0, 0, 1), (9, 7, 1), (0, 3, 3), (0, 3, 5), (9, 4, 5), (0, 0, 6), (0, 3, 6), (9, 4, 6), (0, 0, 7), (0, 3, 7), (9, 7, 8), (0, 3, 9), (9, 4, 11), (0, 6, 12), (9, 7, 12)]
    end = [raw_map.rand_end() for _ in range(len(start))]'''
    '''start = [(9,1,0),(0,0,0)]
    end = [(3,5),(6,2)]'''

    # 贪婪求解
    file_path = "CAsimulator\instance\s_1.xlsx"
    map_path = "map.xlsx"
    sku_data = load_data(file_path,sheet_name="sku_info")
    sku_info = np.array(sku_data.iloc[:,1:])
    demand = np.array(load_data(file_path,sheet_name="demand"))
    map = raw_map.map
    entrances = [i for i in zip(np.where(map==2)[0],np.where(map==2)[1])]
    deliverPos = [i for i in zip(np.where(map==4)[0],np.where(map==4)[1])]
    entrances_dic = {i:entrances[i] for i in range(len(entrances))}
    deliverPos_dic = {i:deliverPos[i] for i in range(len(deliverPos))}
    ST_Table = np.expand_dims(map, axis=0)
    # 计算算法运行时间
    start_time = time.time()
    solution, demand_m_greedy, ST_Table_greedy, time_greedy, path_greedy = greedy_search(map, sku_info,demand,entrances_dic,deliverPos_dic,ST_Table) 
    end_time = time.time()
    print('贪婪分配运行时间：',end_time-start_time)
    print(entrances_dic)
    print(deliverPos_dic)
    print('贪婪分配结果：',solution)
    print('贪婪分配总用时：',time_greedy)
    print(ST_Table_greedy.shape)
    threeD_path(path_greedy)
    print("-"*150+"\n")


    # 随机分配求解
    start = []
    end = []
    demand_random = copy.deepcopy(demand)
    time_random = 0
    for sku in range(len(sku_info)):
        start.append(entrances_dic[sku_info[sku][2]] + (sku_info[sku][1],))
        sku_type = sku_info[sku][0]
        sku_demand = demand[:,sku_type]
        sku_demand_idx = np.where(sku_demand>0)[0]
        random_pos = random.choice(sku_demand_idx)
        end.append(deliverPos_dic[random_pos])
        demand_random[random_pos][sku_type] -= 1
    # print('地图：\n',raw_map.map)
    print('出发点：',start)
    print('终点：',end)
    path = []
    # 初始化时空预约表
    ST_Table_random = np.expand_dims(raw_map.map, axis=0)
    for i in range(len(start)):
        road, work_time=StAstar(raw_map.map,start[i],end[i],ST_Table_random)
        print(f'商品{i}最短路径：',road,'用时：',work_time+1)
        time_random += (work_time+1) 
        ST_Table_random = update_StTable(raw_map.map,ST_Table_random,road[:-1])
        #visualize_map(raw_map.map, start[i], end[i], road)
        path.append(road)
    print('随机分配总用时：',time_random)
    print(ST_Table_random.shape)
    threeD_path(path)
    
    
    