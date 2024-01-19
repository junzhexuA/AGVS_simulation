from map import *
from stA_star import StAstar
from stA_star import update_StTable
from stA_star import find_nearest_exit
from GoodsArrival import *
import random
import numpy as np

random.seed(42)

if __name__ == '__main__':
    raw_map = Map()

    '''#生成到达商品序列
    arrival_sequences = simulate_arrival(6, [100, 150, 200], 0.25, 60)
    #转化成起点信息
    start = covert2start(arrival_sequences)'''

    '''start = [(0, 3, 5), (9, 4, 5), (0, 0, 6), (0, 3, 6)]
    end = [raw_map.rand_end() for _ in range(len(start))]'''

    start = [(0,0,3)]
    end = [(6,2)]
    print('地图：\n',raw_map.map)
    #print('出发点：',start)
    #print('终点：',end)
    transpath = []
    exitpath = []
    path = []
    # 初始化时空预约表
    ST_Table = np.expand_dims(raw_map.map, axis=0)
    
    #AGV搬运路径规划
    for i in range(len(start)):
        road=StAstar(raw_map.map,start[i],end[i],ST_Table)
        #print(f'AGV{i}搬运路径：',road) 
        ST_Table = update_StTable(raw_map.map,ST_Table,road)
        transpath.append(road)

    #形成新的起点，形成随机终点，规划AGV离开路径    
    start_new = [(transpath[i][-1][0],transpath[i][-1][1],transpath[i][-1][2]+1) for i in range(len(transpath))]
    end_new = [find_nearest_exit(raw_map.map,start_new[i]) for i in range(len(transpath))]
    print('出发点：',start_new)
    #print('出口：',end_new)
    ST_Table = np.expand_dims(raw_map.map, axis=0)
    for i in range(len(start_new)):
        exit_road = StAstar(raw_map.map,start_new[i],end_new[i],ST_Table)
        ST_Table = update_StTable(raw_map.map,ST_Table,exit_road)
        exit_road.append((end_new[i][0], end_new[i][1], exit_road[-1][-1]+1))
        exit_road.append(end_new[i])
        #print(f'AGV{i}离开路径：', exit_road)
        exitpath.append(exit_road)

    #合并路径
    for i in range(len(start)):
        path.append(transpath[i]+exitpath[i])
        print(f'AGV{i}路径：', path[i])
        visualize_map(raw_map.map, path[i][0], path[i][-1], path[i])
    #3D可视化
    #threeD_path(path)