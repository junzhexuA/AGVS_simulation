from map import *
from stA_star import StAstar
from stA_star import update_StTable
from GoodsArrival import *
import random
import numpy as np

#random.seed(42)

if __name__ == '__main__':
    raw_map = Map()

    #生成到达商品序列
    #arrival_sequences = simulate_arrival(6, [100, 150, 200], 0.25, 60)
    #转化成起点信息
    #start = covert2start(arrival_sequences)
    #start = [(9, 4, 0), (9, 7, 0), (0, 0, 1), (9, 7, 1), (0, 3, 3), (0, 3, 5), (9, 4, 5), (0, 0, 6), (0, 3, 6), (9, 4, 6), (0, 0, 7), (0, 3, 7), (9, 7, 8), (0, 3, 9), (9, 4, 11), (0, 6, 12), (9, 7, 12), (0, 6, 13), (9, 1, 13), (9, 7, 14), (0, 6, 15), (0, 6, 16), (9, 1, 16), (0, 6, 17), (9, 4, 17), (9, 7, 17), (0, 6, 18), (0, 0, 19), (9, 4, 20), (0, 0, 21), (0, 0, 22), (9, 4, 22), (9, 4, 23), (0, 6, 24), (9, 7, 24), (0, 3, 25), (0, 0, 26), (9, 1, 26), (9, 4, 26), (9, 4, 27), (0, 3, 28), (0, 6, 29), (9, 7, 29), (0, 0, 30), (9, 1, 30), (0, 0, 31), (0, 3, 31), (9, 1, 31), (0, 3, 32), (0, 0, 35), (9, 1, 36), (0, 0, 37), (0, 0, 39), (9, 1, 39), (9, 4, 39), (0, 6, 42), (9, 4, 45), (9, 4, 46), (9, 7, 46), (0, 3, 47), (9, 4, 47), (0, 3, 48), (0, 6, 48), (0, 3, 49), (9, 1, 49), (9, 7, 49), (0, 3, 50), (9, 4, 50), (9, 1, 52), (9, 4, 52), (9, 7, 52), (9, 1, 53), (9, 7, 53), (0, 0, 54), (0, 0, 55), (0, 0, 57), (0, 6, 57), (9, 7, 57), (0, 0, 59)]
    #end = [raw_map.rand_end() for _ in range(len(start))]
    start = [(0,0,32),(0, 3, 35)]
    end = [(3, 5),(3, 5)]
    print('地图：\n',raw_map.map)
    print('出发点：',start)
    print('终点：',end)
    path = []
    # 初始化时空预约表
    ST_Table = np.expand_dims(raw_map.map, axis=0)
    for i in range(len(start)):
        road, work_time=StAstar(raw_map.map,start[i],end[i],ST_Table)
        print(f'商品{i}最短路径：',road,'用时：',work_time+1) 
        ST_Table = update_StTable(raw_map.map,ST_Table,road[:-1])
        visualize_map(raw_map.map, start[i], end[i], road)
        path.append(road)

    threeD_path(path)
    
    