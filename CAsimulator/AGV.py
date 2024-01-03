
# 定义车辆类
class AGV:
    def __init__(self, position, speed=0, needs_turn=False, path=None, state=1,direction=None, entrance=None, destination=None, exit=None, arrival_time=None):
        self.position = position
        self.speed = speed
        self.needs_turn = needs_turn
        self.path = path[1:]
        self.state = state # 1:负载 0:空载
        self.direction = direction # (1,0)向下 (-1,0)向上 (0,1)向右 (0,-1)向左
        self.V_max = 1 # 最大速度
        self.entrance = entrance # agv到达的入口
        self.destination = destination # agv分拣的终点(停靠点)
        self.exit = exit # agv到达的出口
        self.arrival_time = arrival_time #agv开始搬运的时间步
        self.agv_state_value = [-1,-2]

        # 初始化方向, 小型地图
        if position[0] in [0,9] and position[1] in [0,3,6]:
            self.direction = (1,0)
        if position[0] in [0,9] and position[1] in [1,4,7]:
            self.direction = (-1,0)
        
        # 初始化方向, 大型地图
        '''if position[0] in [0,27] and position[1] in [0,3,6,9,12,15,18,21,24,27,30]:
            self.direction = (1,0)
        if position[0] in [0,27] and position[1] in [1,4,7,10,13,16,19,22,25,28,31]:
            self.direction = (-1,0)'''

    def update_speed(self, matrix):
        # 加速规则
        self.speed = min(self.V_max, self.speed + 1)

        # 减速规则,防止碰撞
        dn = self.calculate_dn(matrix)
        self.speed = min(self.speed, dn)


    def calculate_dn(self, matrix):
        # 计算前方空闲元胞数量
        x, y = self.position
        dn = 0
        if self.direction == (1,0):
            for i in range(x+1, matrix.shape[0]):
                if matrix[i][y] not in self.agv_state_value:
                    dn += 1
                else:
                    break
        if self.direction == (-1,0):
            for i in range(x-1, -1, -1):
                if matrix[i][y] not in self.agv_state_value:
                    dn += 1
                else:
                    break
        if self.direction == (0,1):
            for i in range(y+1, matrix.shape[1]):
                if matrix[x][i] not in self.agv_state_value:
                    dn += 1
                else:
                    break
        if self.direction == (0,-1):
            for i in range(y-1, -1, -1):
                if matrix[x][i] not in self.agv_state_value:
                    dn += 1
                else:
                    break
        #print("前方空闲元胞数量:",dn)
        return dn
    
    def update_state(self):
        # 更新车辆状态
        if self.position == self.destination:
            self.state = 0
            self.speed = 0

    def update_position(self, matrix):
        # 更新车辆位置
        if not self.needs_turn:
            if self.speed > 0:
                if len(self.path) > 0:
                    matrix[self.position[0]][self.position[1]] = 0
                    self.position = ((self.position[0] + self.speed * self.direction[0]), (self.position[1] + self.speed * self.direction[1]))
                    if self.state == 1:
                        matrix[self.position[0]][self.position[1]] = -1 # 负载
                        self.path = self.path[1:]
                    if self.state == 0:
                        matrix[self.position[0]][self.position[1]] = -2 # 空载
                        self.path = self.path[1:]
                else:
                    matrix[self.position[0]][self.position[1]] = 0
                    print("AGV已到达目的地")
            if self.speed == 0:
                if len(self.path) > 0:
                    self.position = ((self.position[0] + self.speed * self.direction[0]), (self.position[1] + self.speed * self.direction[1]))
                    if self.state == 1:
                        matrix[self.position[0]][self.position[1]] = -1 # 负载
                    if self.state == 0:
                        matrix[self.position[0]][self.position[1]] = -2 # 空载
                else:
                    matrix[self.position[0]][self.position[1]] = 0             
        # 需要转弯，原地停留
        else:
            self.position = self.position
            self.direction = ((self.path[0][0] - self.position[0]),(self.path[0][1] - self.position[1]))
            self.needs_turn = False
            

    def check_turn(self):
        # 检查是否需要转向（根据实际情况实现）,agv到达终点时不需要转向
        if len(self.path) > 0:
            if ((self.path[0][0] - self.position[0]),(self.path[0][1] - self.position[1])) != self.direction:
                self.needs_turn = True
        else:
            self.needs_turn = False
        

'''# 初始化车辆
agvs = [AGV(full_path1[0], 0, False, full_path1, 1), AGV(full_path2[0], 0, False, full_path2, 1)]'''

# 模拟单步
def simulate_step(matrix, agvs):
    for agv in agvs:
        agv.update_speed(matrix)
        agv.check_turn()
        if agv.state == 1:
            agv.update_state()
        agv.update_position(matrix)
        

    return matrix, agvs

'''# 定义最大速度
V_max = 1
# 定义10*8的路网矩阵
matrix = np.full((10,8), 0)

# 时间步为0初始化AGV信息
for j in range(len(agvs)):
    matrix[agvs[j].position[0]][agvs[j].position[1]] = -1
    #print(f"AGV{j}当前位置：",agvs[j].position,f"AGV{j}当前速度：",agvs[j].speed,f"AGV{j}路径：",agvs[j].path)
print("第",0,"步")
print(matrix)
   

# 运行模拟
time_Step = 12
for i in range(time_Step):
    print("第",i+1,"步")
    matrix, agvs = simulate_step(matrix, agvs)
    for j in range(len(agvs)):
        #输出AGV1的信息      
        print(f"AGV{j}当前位置：",agvs[j].position,f"AGV{j}当前速度：",agvs[j].speed,f"AGV{j}路径：",agvs[j].path)
    print(matrix)'''