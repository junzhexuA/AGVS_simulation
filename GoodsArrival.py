import numpy as np

def simulate_arrival(n, total_goods, lambda_, duration):
    # 商品种类
    goods_type = [1, 2, 3]

    # 投递口数量
    n = n

    # 模拟duration秒
    arrival_sequences = []

    for i in range(n):
        arrival_sequence = []
        gi = i
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
                if total_goods[goods_index] < 0:
                    arrival_num = total_goods[goods_index]
                    total_goods[goods_index] = 0
                    goods_type.remove(goods_index)
                    print(f"商品{total_goods[goods_index]}搬运任务以完成")
                    
                # 添加到到达序列
                arrival_sequence.append(goods_type[goods_index])
            else:
                # 如果没有商品到达，添加0到到达序列
                arrival_sequence.append(goods_type[-1])
        for ti, k in enumerate(arrival_sequence):
            if k>0:
                arrival_sequences.append([gi,ti,i])


    return arrival_sequences

def covert2start(arrival_sequences):
    # 定义起点字典
    start_points = {0: (0, 0), 1: (0, 3), 2: (0, 6), 3: (9, 1), 4: (9, 4), 5: (9, 7)}

    # 转换格式并排序
    starts = [(start_points[item[0]][0], start_points[item[0]][1], item[1]) for item in arrival_sequences]
    starts.sort(key=lambda x: x[2])

    return starts

# 测试函数
if __name__ == '__main__':
    n = 6 #入口数量
    total_goods = [10, 15, 20] #商品总数
    lambda_ = 0.25 #到达率，件/秒
    duration = 60 #模拟未来1分钟

    #生成到达商品序列
    arrival_sequences = simulate_arrival(n, total_goods, lambda_, duration)
    starts = covert2start(arrival_sequences)
    print(arrival_sequences)