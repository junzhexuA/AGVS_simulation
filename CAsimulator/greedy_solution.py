import numpy as np
import pandas as pd
import copy
from SpaceTimeA_star import *

def load_data(file_path,sheet_name):
    data =pd.read_excel(file_path,sheet_name=sheet_name,header=None)
    return data[1:]

def load_map(map_path):
    data =pd.read_excel(map_path,header=None)
    map = np.array(data)
    return map

# 贪婪搜索，根据sku的到达入口，找到有需求的最近的投递口
def greedy_search(map,sku_info, demand, entrances_dic, deliverPos_dic,ST_Table):
    solution = []
    path = []
    time = 0
    demand = copy.deepcopy(demand)
    for sku in range(len(sku_info)):
        sku_pos = entrances_dic[sku_info[sku][2]]
        sku_type = sku_info[sku][0]
        sku_demand = demand[:,sku_type]
        # 返回有需求的投递口
        sku_demand_idx = np.where(sku_demand>0)[0]
        # 计算距离
        time2deliverPos = {}
        tmp = copy.deepcopy(ST_Table)
        for i in sku_demand_idx:
            road, time2deliverPos[i] = StAstar(map, (sku_pos + (sku_info[sku][1],)), deliverPos_dic[i], tmp)
            tmp = copy.deepcopy(ST_Table)  
        # 返回最近的投递口
        best_pos = min(time2deliverPos, key=time2deliverPos.get)
        road, time2deliver = StAstar(map, (sku_pos + (sku_info[sku][1],)), deliverPos_dic[best_pos], ST_Table)
        print(f'商品{sku}最短路径：',road,'用时：',time2deliver+1)
        ST_Table = update_StTable(map,ST_Table,road[:-1])
        time += (time2deliver+1)
        path.append(road)
        # 更新需求
        demand[best_pos][sku_type] -= 1
        solution.append(best_pos)
    return solution, demand, ST_Table, time, path


'''if __name__ == '__main__':
    file_path = "CAsimulator\instance\s_1.xlsx"
    map_path = "map.xlsx"
    sku_data = load_data(file_path,sheet_name="sku_info")
    sku_info = np.array(sku_data.iloc[:,1:])
    demand = np.array(load_data(file_path,sheet_name="demand"))
    map = load_map(map_path)
    entrancePos = [i for i in zip(np.where(map==2)[0],np.where(map==2)[1])]
    deliverPos = [i for i in zip(np.where(map==4)[0],np.where(map==4)[1])]
    entrances_dic = {i:entrancePos[i] for i in range(len(entrancePos))}
    deliverPos_dic = {i:deliverPos[i] for i in range(len(deliverPos))}
    ST_Table = np.expand_dims(map, axis=0)
    solution, demand_new = greedy_search(map,sku_info,demand,entrances_dic,deliverPos_dic)
    total_time = calculate_time(solution,sku_info,entrances_dic,deliverPos_dic,map)
    print('总用时：',total_time)'''

