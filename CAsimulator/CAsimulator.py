from AGV import simulate_step
from Task import task

def simulate(frame, ax, agv_matrix):
    global agvs, tmp

    print("第", frame, "步")
    agv_matrix, agvs = simulate_step(agv_matrix, agvs)

    # Output AGV information
    for j in range(len(agvs)):
        print(f"AGV{j}当前位置：", agvs[j].position, f"AGV{j}当前速度：", agvs[j].speed, f"AGV{j}路径：", agvs[j].path)

    for j in tmp:
        if j == frame:
            print(f"第{j}时间步有新任务到达")
            agvs.append(AGV(task_list[arrival_time_list.index(j)].full_path[0], 0, False, task_list[arrival_time_list.index(j)].full_path, 1, direction=None, 
                            entrance=task_list[arrival_time_list.index(j)].entrance, destination=task_list[arrival_time_list.index(j)].destination, 
                            exit=task_list[arrival_time_list.index(j)].exit, arrival_time=task_list[arrival_time_list.index(j)].arrival_time))
            tmp.remove(j)
            for k in range(len(agvs)):
                if len(agvs[k].path) > 0:
                    agv_matrix[agvs[k].position[0]][agvs[k].position[1]] = -1
                # Output AGV information
                print(f"AGV{k}当前位置：", agvs[k].position, f"AGV{k}当前速度：", agvs[k].speed, f"AGV{k}起点：", 
                      agvs[k].entrance, f"AGV{k}投递口：", agvs[k].destination, f"AGV{k}出口：", agvs[k].exit, 
                      f"AGV{k}任务开始时间步：", agvs[k].arrival_time)

    print(agv_matrix)
    img = ax.imshow(agv_matrix, cmap='gray_r', animated=True)  # Redraw the image
    ax.grid()
    return img,