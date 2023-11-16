from map import *
from stA_star import StAstar
from stA_star import update_StTable
from GoodsArrival import *
import random
import numpy as np

random.seed(42)

if __name__ == '__main__':
    raw_map = Map()

    '''#生成到达商品序列
    arrival_sequences = simulate_arrival(6, [100, 150, 200], 0.05, 60)
    #转化成起点信息
    start = covert2start(arrival_sequences)
    end = [raw_map.rand_end() for _ in range(len(start))]'''
    start = [(0,0,1),(9,1,8),(9,7,7),(9,4,10)]
    end = [(6,5),(3,2),(6,2),(6,2)]
    print('地图：\n',raw_map.map)
    print('出发点：',start)
    print('终点：',end)
    path = []
    # 初始化时空预约表
    ST_Table = np.expand_dims(raw_map.map, axis=0)
    for i in range(len(start)):
        road=StAstar(raw_map.map,start[i],end[i],ST_Table)
        print('A*  最短路径：',road) 
        ST_Table = update_StTable(raw_map.map,ST_Table,road[:-1])
        # visualize_map(raw_map.map, start[i], end[i], road)
        path.append(road)

    threeD_path(path)
    
    