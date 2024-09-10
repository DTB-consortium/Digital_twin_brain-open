import numpy as np
import time
import matplotlib.pyplot as plt

def generate_sequential_map(N, voxels_per_dcu):
    map_table = list()
    for i in range(N):
        map_table.append([])

    cnt, idx = 0, 0
    for key in voxels_per_dcu:
        for i in range(voxels_per_dcu[key]):
            for j in range(key):
                map_table[idx].append(cnt)
                cnt += 1
            idx += 1
    return map_table

def cal_size_multi_degree_conn_split(N, map_table,size,degree,conn):
    assert type(map_table[0]) == dict
    size_degree = np.zeros(N)
    for gpu_idx in range(N):
        for population_idx in map_table[gpu_idx].keys():
            size_degree[gpu_idx] += float(map_table[gpu_idx][population_idx] * size[population_idx] * degree[population_idx] * max(0.001,conn[population_idx]))

    return size_degree


def cal_size_multi_degree_split(N, map_table, size, degree):
    assert type(map_table[0]) == dict
    size_degree = np.zeros(N)
    for gpu_idx in range(N):
        for population_idx in map_table[gpu_idx].keys():
            size_degree[gpu_idx] += float(map_table[gpu_idx][population_idx] * size[population_idx] * degree[population_idx])

    return size_degree


def cal_size_split(N, map_table, size_table):
    assert type(map_table[0]) == dict
    size_degree = np.zeros(N)
    for gpu_idx in range(N):
        for population_idx in map_table[gpu_idx].keys():
            size_degree[gpu_idx] += map_table[gpu_idx][population_idx] * size_table[population_idx]
    return size_degree

def map_table_normal_to_split(N, map_table):
    map_table_split = list()
    for i in range(N):
        map_table_split.append(dict())

    # 生成按标号顺序将标号顺序映射到GPU的映射表，此时每个体素的初始部分都为1
    for gpu_idx in range(N):
        for population_idx in map_table[gpu_idx]:
            map_table_split[gpu_idx][population_idx] = 1

    return map_table_split

def generate_map_split_only_size(M, N, size_table, degree_table ,max_rate, max_iter_times, only_size):
        """
        生成分割后的地图函数，根据输入参数进行 GPU 的分配调整和流量调整。

        参数:
            M: int
                体素/功能柱的数量
            N： int
                卡数
            max_rate: float
                最大值和平均值的比值，用于控制分割后 GPU 的大小差异。
            max_iter_times: int
                最大迭代次数，用于控制算法收敛的迭代次数上限。
            k_traffic: int
                流量调整的权重系数。
            k_split: int
                分割调整的权重系数。。
            only_size: bool
                是否只根据size进行划分，默认true；如果选择false，则同时根据size和degree进行划分

        返回值:
            pickle格式的map表

        """
        voxels_per_dcu = {M//N:N-(M%N), M//N+1:M%N}
        sequential_map_without_invalid_index = generate_sequential_map(N, voxels_per_dcu)

        # 计算出连续地图中每个 GPU 的最大度量值
        max_i = 0
        for i in range(N):
            max_tmp = max(sequential_map_without_invalid_index[i])
            max_i = max(max_i, max_tmp)
        print(max_i)

        # 将连续地图映射为分割地图，并计算分割地图中每个 GPU 的最大度量值
        map_table_split = map_table_normal_to_split(N, sequential_map_without_invalid_index)
        max_i = 0
        print("----------")
        for i in range(N):
            max_tmp = max(map_table_split[i])
            max_i = max(max_i, max_tmp)
        print(max_i)

        # 初始化 GPU 的分割地1
        map_table = map_table_split

        #只以size作为平衡标准
        if only_size:
            print("only size")
            origin_size = cal_size_split(N, map_table, size_table)

            # 计时开始，用于评估算法运行时间
            time1 = time.time()
            print('Begin to generate map...')

            # 初始化一些参数
            size_per_gpu = np.array(origin_size)  # deep copy
            average = np.average(size_per_gpu)
            cnt = 0
            best_obj = 999

            # 进行 GPU 的大小调整和流量调整，直到满足停止条件或达到最大迭代次数
            while np.max(size_per_gpu) > np.average(size_per_gpu) * max_rate and cnt < max_iter_times:
                best_obj = min(best_obj, np.max(size_per_gpu) / average)

                if cnt % 1000 == 0:
                    print('iter %d: best_obj: %.8f, target: %.4f' % (cnt, best_obj, max_rate))

                cnt += 1

                # 随机选出大小 * degree 小/大的 GPU，并返回其编号
                copy_size_per_dcu = size_per_gpu.copy()
                copy_size_per_dcu.sort()
                max_idx_temp = np.random.randint(1, 2)
                min_idx_temp = np.random.randint(1, 20)
                max_size_gpu_idx = np.where(size_per_gpu == copy_size_per_dcu[-max_idx_temp])[0][0]
                min_size_gpu_idx = np.where(size_per_gpu == copy_size_per_dcu[min_idx_temp - 1])[0][0]

                # 随机选出流量大的 population，并得到其绝对编号，对于已经拆分的 population 不选择
                # 选择一个 GPU 中流量大的 population，用于后续交换
                size_per_population_inside_gpu = dict()
                for population_idx in map_table[max_size_gpu_idx].keys():
                    if map_table[max_size_gpu_idx][population_idx] == 1:  # population 未被拆分
                        size_per_population_inside_gpu[population_idx] = size_table[population_idx] * \
                                                                         map_table[max_size_gpu_idx][population_idx]
                    else:  # population 在预处理阶段已经被拆分
                        size_per_population_inside_gpu[population_idx] = 0  # 对于已经拆分的 population 记为0，以确保不会被选择
                temp2 = sorted(size_per_population_inside_gpu.items(), key=lambda x: x[1], reverse=True)
                idx_temp = np.random.randint(0, 3)
                max_voxel_absolute_idx = temp2[idx_temp][0]

                # 随机选出流量小的 population，并得到其绝对编号，对于已经拆分的 population 不选择
                # 选择一个 GPU 中流量小的 population，用于后续交换
                size_per_population_inside_gpu = dict()
                for population_idx in map_table[min_size_gpu_idx].keys():  # population 未被拆分
                    if map_table[min_size_gpu_idx][population_idx] == 1:
                        size_per_population_inside_gpu[population_idx] = size_table[population_idx] * \
                                                                         map_table[min_size_gpu_idx][population_idx]
                    else:  # population 在预处理阶段已经被拆分
                        size_per_population_inside_gpu[population_idx] = 1e3  # 对于已经拆分的 population 记为很大的数，以确保不会被选择

                temp2 = sorted(size_per_population_inside_gpu.items(), key=lambda x: x[1], reverse=False)
                idx_temp = np.random.randint(0, 1)
                min_voxel_absolute_idx = temp2[idx_temp][0]

                # 选出的大流量 population 与小流量 population 互换位置，进行交换
                max_key, min_key = max_voxel_absolute_idx, min_voxel_absolute_idx
                map_table[max_size_gpu_idx][min_key] = map_table[max_size_gpu_idx].pop(max_key)
                map_table[min_size_gpu_idx][max_key] = map_table[min_size_gpu_idx].pop(min_key)

                # 更新 size * degree
                size_per_gpu[max_size_gpu_idx] = size_per_gpu[max_size_gpu_idx] - size_table[
                    max_voxel_absolute_idx] + size_table[min_voxel_absolute_idx]
                size_per_gpu[min_size_gpu_idx] = size_per_gpu[min_size_gpu_idx] + size_table[
                    max_voxel_absolute_idx] - size_table[min_voxel_absolute_idx]

            # 计时结束，输出结果
            time2 = time.time()
            print('Map generated. Iter times: %d, %.2fs consumed' % (cnt, (time2 - time1)))

            best_obj = min(best_obj, np.max(size_per_gpu) / np.average(size_per_gpu))
            print('best_obj: %.6f, target: %.6f' % (best_obj, max_rate))

            # 计算最终 GPU 的大小和度量值，存储在 ultimate_size_degree 列表中
            ultimate_size_degree = cal_size_split(N, map_table, size_table)
            print('Check Size Degree: %.6f' % (np.max(ultimate_size_degree) / np.average(ultimate_size_degree)))

            # 确保map_table是一个列表，每个元素是一个小字典
            if not isinstance(map_table, list) or not all(isinstance(item, dict) for item in map_table):
                raise ValueError("The .npy file should contain a list of dictionaries.")

            # 将data转换回NumPy数组（可选，如果需要继续使用NumPy数组）
            map_table = np.array(map_table)

            # 将NumPy数组转换为嵌套字典
            nested_dict = {}
            for i, small_dict in enumerate(map_table):
                nested_dict[str(i)] = [key for key in small_dict.keys()]

            print('map is generated!')
            return nested_dict

        else:
            print("both size and degree")
            table = [a2 * a3 for a2, a3 in zip(degree_table, size_table)]
            origin_size = cal_size_multi_degree_split(N, map_table, size_table, degree_table)

            # 计时开始，用于评估算法运行时间
            time1 = time.time()
            print('Begin to generate map...')

            # 初始化一些参数
            size_per_gpu = np.array(origin_size)  # deep copy
            average = np.average(size_per_gpu)
            cnt = 0
            best_obj = 999

            # 进行 GPU 的大小调整和流量调整，直到满足停止条件或达到最大迭代次数
            while np.max(size_per_gpu) > np.average(size_per_gpu) * max_rate and cnt < max_iter_times:
                best_obj = min(best_obj, np.max(size_per_gpu) / average)

                if cnt % 1000 == 0:
                    print('iter %d: best_obj: %.8f, target: %.4f' % (cnt, best_obj, max_rate))

                cnt += 1

                # 随机选出大小 * degree 小/大的 GPU，并返回其编号
                copy_size_per_dcu = size_per_gpu.copy()
                copy_size_per_dcu.sort()
                max_idx_temp = np.random.randint(1, 2)
                min_idx_temp = np.random.randint(1, 19)
                max_size_gpu_idx = np.where(size_per_gpu == copy_size_per_dcu[-max_idx_temp])[0][0]
                min_size_gpu_idx = np.where(size_per_gpu == copy_size_per_dcu[min_idx_temp - 1])[0][0]

                # 随机选出流量大的 population，并得到其绝对编号，对于已经拆分的 population 不选择
                # 选择一个 GPU 中流量大的 population，用于后续交换
                size_per_population_inside_gpu = dict()
                for population_idx in map_table[max_size_gpu_idx].keys():
                    if map_table[max_size_gpu_idx][population_idx] == 1:  # population 未被拆分
                        size_per_population_inside_gpu[population_idx] = table[population_idx]*\
                                                                         map_table[max_size_gpu_idx][population_idx]
                    else:  # population 在预处理阶段已经被拆分
                        size_per_population_inside_gpu[population_idx] = 0  # 对于已经拆分的 population 记为0，以确保不会被选择
                temp2 = sorted(size_per_population_inside_gpu.items(), key=lambda x: x[1], reverse=True)
                idx_temp = np.random.randint(0, 3)
                max_voxel_absolute_idx = temp2[idx_temp][0]

                # 随机选出流量小的 population，并得到其绝对编号，对于已经拆分的 population 不选择
                # 选择一个 GPU 中流量小的 population，用于后续交换
                size_per_population_inside_gpu = dict()
                for population_idx in map_table[min_size_gpu_idx].keys():  # population 未被拆分
                    if map_table[min_size_gpu_idx][population_idx] == 1:
                        size_per_population_inside_gpu[population_idx] = table[population_idx] *\
                                                                         map_table[min_size_gpu_idx][population_idx]
                    else:  # population 在预处理阶段已经被拆分
                        size_per_population_inside_gpu[population_idx] = 1e3  # 对于已经拆分的 population 记为很大的数，以确保不会被选择

                temp2 = sorted(size_per_population_inside_gpu.items(), key=lambda x: x[1], reverse=False)
                idx_temp = np.random.randint(0, 1)
                min_voxel_absolute_idx = temp2[idx_temp][0]

                # 选出的大流量 population 与小流量 population 互换位置，进行交换
                max_key, min_key = max_voxel_absolute_idx, min_voxel_absolute_idx

                #print(max_voxel_absolute_idx,min_voxel_absolute_idx)

                map_table[max_size_gpu_idx][min_key] = map_table[max_size_gpu_idx].pop(max_key)
                map_table[min_size_gpu_idx][max_key] = map_table[min_size_gpu_idx].pop(min_key)

                #print('max_gpu',max_size_gpu_idx,min_key)
                #print('min_gpu',min_size_gpu_idx,max_key)

                # 更新 size * degree
                size_per_gpu[max_size_gpu_idx] = size_per_gpu[max_size_gpu_idx] - table[max_voxel_absolute_idx] \
                                                 + table[min_voxel_absolute_idx]
                size_per_gpu[min_size_gpu_idx] = size_per_gpu[min_size_gpu_idx] + table[max_voxel_absolute_idx] \
                                                 - table[min_voxel_absolute_idx]

            # 计时结束，输出结果
            time2 = time.time()
            print('Map generated. Iter times: %d, %.2fs consumed' % (cnt, (time2 - time1)))

            best_obj = min(best_obj, np.max(size_per_gpu) / np.average(size_per_gpu))
            print('best_obj: %.6f, target: %.6f' % (best_obj, max_rate))

            # 计算最终 GPU 的大小和度量值，存储在 ultimate_size_degree 列表中
            ultimate_size_degree = cal_size_multi_degree_split(N, map_table, size_table, degree_table)
            print('Check Size Degree: %.6f' % (np.max(ultimate_size_degree) / np.average(ultimate_size_degree)))

            # 确保map_table是一个列表，每个元素是一个小字典
            if not isinstance(map_table, list) or not all(isinstance(item, dict) for item in map_table):
                raise ValueError("The .npy file should contain a list of dictionaries.")

            # 将data转换回NumPy数组（可选，如果需要继续使用NumPy数组）
            map_table = np.array(map_table)
            all_inner_keys = {key for inner_dict in map_table for key in inner_dict.keys()}
            # 将NumPy数组转换为嵌套字典
            nested_dict = {}
            for i, small_dict in enumerate(map_table):
                nested_dict[str(i)] = [key for key in small_dict.keys()]
            
            print('map is generated!')
            return nested_dict
 
