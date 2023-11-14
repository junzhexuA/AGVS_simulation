import heapq
import numpy as np

def heuristic(a, b, array,raw_orientation=None,new_orientation=None):
    """
    计算两个点之间的曼哈顿距离
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_neighbors(point,array=None):
    neighbors=[]
    if point[0] in [1,4,7]:
        neighbors.append((0,1))
    if point[0] in [2,5,8]:
        neighbors.append((0,-1))
    if point[1] in [0,3,6]:
        neighbors.append((1,0))
    if point[1] in [1,4,7]:
        neighbors.append((-1,0))
    return neighbors

#好像存在bug
def update_StTable(raw_map, StTable, path):
    if len(StTable) - 1 < path[-1][2]:
        new_time_step = path[-1][2] - len(StTable) + 1
        new_time_step_map = np.repeat(np.expand_dims(raw_map,axis=0), new_time_step, axis=0)
        for node in path[len(path) - new_time_step :]:
            new_time_step_map[node[2]-len(StTable)][node[0]][node[1]] = 6
        StTable = np.concatenate([StTable,new_time_step_map],axis=0)       
        for node in path[:len(path) - new_time_step]:
            StTable[node[2]][node[0]][node[1]] = 6
    else:
        for node in path:
            StTable[node[2]][node[0]][node[1]] = 6
    return StTable

def Astar(array, start, goal, StTable):
    """
    A* 寻路算法
    """

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal, array)}
    oheap = []
    #初始化方向
    last_orientation=(0,0)


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
                data.append(goal)
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                return data[::-1]
        # 判断该点可达的邻居
        neighbors=find_neighbors(current,array)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j, current[2] + 1  #这个地方有BUG，这里默认到达邻居的时间步是+1，但是如果需要转弯才能到达这里应该+2
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    # 判断是否为障碍物 此处可以改成其他限制条件
                    if array[neighbor[0]][neighbor[1]] == 5:
                        continue
                    if array[neighbor[0]][neighbor[1]] == 4 and neighbor != goal:
                        continue
                    #判断是否为动态障碍物，即优先级高的AGV在该时间步的位置
                    if neighbor[2]+1 <= len(StTable):
                        if StTable[neighbor[2]][neighbor[0]][neighbor[1]] == 6:
                            continue                 
                else:
                    # 超出边界
                    continue
            else:
                # 超出边界
                continue
            # 判断是否转向
            if current in came_from.keys():
                last_orientation=(current[0]-came_from[current][0],current[1]-came_from[current][1])
            # 如果要转向，距离额外加1
            if last_orientation!=(i,j):
                tentative_g_score = gscore[current] + heuristic(current, neighbor,array)+1
            else:
                tentative_g_score = gscore[current] + heuristic(current, neighbor,array)
            # 如果距离更远，排除
            if neighbor in close_set  and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            # 如果距离更近，更新
            if  tentative_g_score < gscore.get(neighbor, 0) or (neighbor not in [i[1]for i in oheap]):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal,array)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                signal=1
        if ~signal:
            tentative_g_score = gscore[current]+1
            current=(current[0] , current[1] , current[2] + 1)
            gscore[current] = tentative_g_score
            fscore[current] = tentative_g_score + heuristic(current, goal,array)
            heapq.heappush(oheap, (fscore[current], current))
    return None
