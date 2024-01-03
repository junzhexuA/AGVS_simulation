import numpy as np

def simulate_arrival(n, total_goods, lambda_, duration):
    # 商品种类
    goods_type = [1, 2, 3, 0]

    # 投递口数量
    n = n

    # 模拟duration秒
    arrival_sequences = []

    for i in range(n):
        arrival_sequence = []
        for second in range(duration):
        # 每秒到达的商品数量
            arrival_num = np.random.poisson(lambda_)
            # 如果有商品到达
            if arrival_num > 0:
                # 随机选择一种商品
                goods_index = np.random.choice([0, 1, 2], p=[total_goods[0]/sum(total_goods), total_goods[1]/sum(total_goods), total_goods[2]/sum(total_goods)])
                # 更新商品总量
                total_goods[goods_index] -= arrival_num
                # 如果商品总量小于0，设置为0
                total_goods[goods_index] = max(0, total_goods[goods_index])
                # 添加到到达序列
                arrival_sequence.append(goods_type[goods_index])
            else:
                # 如果没有商品到达，添加0到到达序列
                arrival_sequence.append(goods_type[-1])
        for ti, j in enumerate(arrival_sequence):
            if j>0:
                arrival_sequences.append([i,ti,j])


    return arrival_sequences

def covert2start(arrival_sequences):
    # 定义起点字典
    start_points = {0: (0, 0), 1: (0, 3), 2: (0, 6), 3: (9, 1), 4: (9, 4), 5: (9, 7)} #小型地图
    '''start_points = {0: (0, 0), 1: (0, 3), 2: (0, 6), 3:(0,9), 4:(0,12), 5:(0,15), 6:(0,18), 7:(0,21), 8:(0,24), 9:(0,27), 10:(0,30),
                    11:(27,1), 12:(27,4), 13:(27,7), 14:(27,10), 15:(27,13), 16:(27,16), 17:(27,19), 18:(27,22), 19:(27,25), 20:(27,28), 21:(27,31)} #大型地图'''

    # 转换格式并排序
    starts = [(start_points[item[0]][0], start_points[item[0]][1], item[1]) for item in arrival_sequences]
    starts.sort(key=lambda x: x[2])

    return starts

# 测试函数
if __name__ == '__main__':
    arrival_sequences = simulate_arrival(6, [100, 150, 200], 0.25, 30)
    starts = covert2start(arrival_sequences)
    print(starts)