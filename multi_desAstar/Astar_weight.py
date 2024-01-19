import heapq
import numpy as np

alpha = 5
beta = 0.5
gamma = 10
def heuristic(a, b, array,raw_orientation=None,new_orientation=None):
    """
    计算两个点之间的曼哈顿距离
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

#搜索map中最近的出口
def find_nearest_exit(map_array, start):
    pos_of_exits = np.argwhere(map_array == 3)
    pos_of_exits.tolist()
    distances = [calculate_distance(start, (x, y)) for x, y in pos_of_exits]
    min_distance = min(distances)
    closest_position = pos_of_exits[distances.index(min_distance)]
    
    return (closest_position[0],closest_position[1])

def update_start(StTable, start):
    if len(StTable) - 1 >= start[2]:
        print(len(StTable))
        # 检查新的start是否已经被使用过
        if is_start_used(StTable, start):
            return update_start(StTable, (start[0], start[1], start[2] + 1))
    return start

def is_start_used(StTable, start):
    # 检查start是否占用
    return StTable[start[2]][start[0]][start[1]] == 6

def find_neighbors(point,array=None):
    neighbors=[]
    if point[0] in [1,4,7,10,13,16,19,22,25]:
        neighbors.append((0,1))
    if point[0] in [2,5,8,11,14,17,20,23,26]:
        neighbors.append((0,-1))
    if point[1] in [0,3,6,9,12,15,18,21,24,27,30]:
        neighbors.append((1,0))
    if point[1] in [1,4,7,10,13,16,19,22,25,28,31]:
        neighbors.append((-1,0))
    return neighbors

def Astar(array, start, goal, network_weight,direction=None):
    """
    A* 寻路算法
    """
    '''if len(StTable) - 1 > start[2]:
        while StTable[start[2]][start[0]][start[1]] == 6:
            start = update_start(StTable,start)'''

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal, array)}
    oheap = []

    #初始化方向，位于上方入口（行索引为0），初始方向向下；位于下方入口，初始方向向上
    if start[0] == 0:
        last_orientation = (1,0)
    if start[0] == len(array)-1:
        last_orientation = (-1,0)
    if start[0] in [2,5,8,11,14,17,20,23]:
        last_orientation = (0,-1)
    if start[0] in [4,7,10,13,16,19,22,25]:
        last_orientation = (0,1)
    if start[1] in [1,4,7,10,13,16,19,22,25,28]:
        last_orientation = (-1,0)
    if start[1] in [3,6,9,12,15,18,21,24,27,30]:
        last_orientation = (1,0)
    else:
        last_orientation = direction
    
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        # 标记一下这次有没有找到路
        signal=0
        current = heapq.heappop(oheap)[1]
        close_set.add(current)
        # 判断是否成功到达卸货单元
        for neighbor in [(0,1),(0,-1),(1,0),(-1,0)]:
            if (current[0]+neighbor[0],current[1]+neighbor[1]) == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                return data[::-1]
        # 判断该点可达的邻居
        neighbors=find_neighbors(current,array)
        for i, j in neighbors:
            # 判断是否转向
            if current in came_from.keys():
               if current[0] != came_from[current][0] or current[1] != came_from[current][1]:
                   last_orientation=(current[0]-came_from[current][0],current[1]-came_from[current][1]) #若停留的话last_orientation会成为(0,0),修改后停留时方向不变 

            neighbor = current[0] + i, current[1] + j 

            # 约束：1.静态障碍物; 2.地图边界； 3.动态障碍
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    # 判断是否为障碍物 此处可以改成其他限制条件
                    if array[neighbor[0]][neighbor[1]] == 5:
                        signal = 0
                        continue
                    if array[neighbor[0]][neighbor[1]] == 4 and neighbor != goal:
                        signal = 0
                        continue
                else:
                    # 超出边界
                    signal = 0
                    continue
            else:
                # 超出边界
                signal = 0
                continue
            
            # 如果要转向，距离额外加1
            if last_orientation!=(i,j):
                #tentative_g_score = gscore[current] + heuristic(current, neighbor, array) + 1 + 1/network_weight[current[0]][current[1]]
                #tentative_g_score = gscore[current] + heuristic(current, neighbor, array) + 1
                tentative_g_score = gscore[current] + heuristic(current, neighbor, array) + 1 + gamma / (1 + np.exp(alpha * (network_weight[current[0]][current[1]] - beta)))
            else:
                #tentative_g_score = gscore[current] + heuristic(current, neighbor, array) + 1/network_weight[current[0]][current[1]]
                #tentative_g_score = gscore[current] + heuristic(current, neighbor, array)
                tentative_g_score = gscore[current] + heuristic(current, neighbor, array) + gamma / (1 + np.exp(alpha * (network_weight[current[0]][current[1]] - beta)))
            # 如果距离更远，排除
            if neighbor in close_set  and tentative_g_score > gscore.get(neighbor, 0):
                signal = 0
                continue
            # 如果距离更近，更新
            if  tentative_g_score <= gscore.get(neighbor, 0) or (neighbor not in [i[1]for i in oheap]):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                #fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal,array) + 1/network_weight[neighbor[0]][neighbor[1]]
                #fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal,array)
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal,array) + gamma /(1 + np.exp(alpha * (network_weight[current[0]][current[1]] - beta)))
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None
