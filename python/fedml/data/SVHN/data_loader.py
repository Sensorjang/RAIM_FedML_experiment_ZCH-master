import json
import os
import math
import numpy as np
import scipy.io as sio
from ...ml.engine import ml_engine_adapter
import logging
import torch


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .mat files
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    # Load train data
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".mat")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        data = sio.loadmat(file_path)
        x = data['X'].transpose((3, 0, 1, 2))  # Convert to NHWC format
        y = data['y'].flatten()
        # Create a client id based on the file name
        client_id = f.split('.')[0]
        clients.append(client_id)
        train_data[client_id] = {'x': x, 'y': y}
        # print(f"Loaded train data for client {client_id}: {len(x)} samples")

    # Load test data
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".mat")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        data = sio.loadmat(file_path)
        x = data['X'].transpose((3, 0, 1, 2))  # Convert to NHWC format
        y = data['y'].flatten()
        # Create a client id based on the file name
        client_id = f.split('.')[0]
        test_data[client_id] = {'x': x, 'y': y}
        # print(f"Loaded test data for client {client_id}: {len(x)} samples")

    clients = sorted(clients)

    return clients, groups, train_data, test_data


def batch_data(args, data, batch_size):
    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]

        # Convert to NCHW format
        batched_x = batched_x.transpose((0, 3, 1, 2))  # NHWC to NCHW

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 转换为 PyTorch Tensor 并移动到指定设备
        batched_x = torch.from_numpy(batched_x).float().to(device)
        batched_y = torch.from_numpy(batched_y).long().to(device)

        batch_data.append((batched_x, batched_y))

    return batch_data


def load_partition_data_svhn_by_device_id(batch_size, device_id, train_path="SVHN_mobile", test_path="SVHN_mobile"):
    train_path += os.path.join("/", device_id, "train")
    test_path += os.path.join("/", device_id, "test")
    return load_partition_data_svhn(batch_size, train_path, test_path)


def load_partition_data_svhn(
    args, batch_size, train_path=os.path.join(os.getcwd(), "SVHN", "train"),
        test_path=os.path.join(os.getcwd(), "SVHN", "test")
):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        # 修改客户端 ID 以匹配测试数据
        test_client_id = u.replace('train', 'test')
        if test_client_id not in test_data:
            # 如果测试数据中没有该客户端的数据，跳过
            # print(f"Client {u} not found in test data, skipping...")
            continue
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[test_client_id]["x"])

        ############RAIM############
        # 获取当前客户端的声誉值
        if client_idx < len(args.reputations):
            client_reputation = args.reputations[client_idx]
        else:
            client_reputation = -0.5  # 默认声誉值
        # 根据声誉值计算折扣因子
        if client_reputation < 0:
            discount_factor = 0.5 + (1 + client_reputation) * 0.5  # 当声誉为负时，折扣从 0.5 到 1.0
        else:
            discount_factor = 1.0  # 当声誉为非负时，折扣为 1.0（即不打折）
        # 应用折扣并四舍五入为整数
        discounted_train_data_num = math.ceil(user_train_data_num * discount_factor)
        ############RAIM############

        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = discounted_train_data_num

        # transform to batches
        train_batch = batch_data(args, train_data[u], batch_size)
        test_batch = batch_data(args, test_data[test_client_id], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        # print(f"Processed client {client_idx}: train data {user_train_data_num}, test data {user_test_data_num}")
        client_idx += 1

    # 根据 args.client_num_in_total 进行随机划分
    if args.client_num_in_total > 1:
        # 获取所有训练数据
        all_train_x = np.concatenate([v["x"] for v in train_data.values()])
        all_train_y = np.concatenate([v["y"] for v in train_data.values()])
        all_test_x = np.concatenate([v["x"] for v in test_data.values()])
        all_test_y = np.concatenate([v["y"] for v in test_data.values()])

        # 随机打乱数据
        np.random.seed(100)
        idx = np.random.permutation(len(all_train_x))
        all_train_x = all_train_x[idx]
        all_train_y = all_train_y[idx]

        idx = np.random.permutation(len(all_test_x))
        all_test_x = all_test_x[idx]
        all_test_y = all_test_y[idx]

        # 生成不平均划分的随机数据量
        np.random.seed(42)  # 设置随机种子以确保可重复性
        proportions = np.random.dirichlet(np.ones(args.client_num_in_total), size=1).flatten()
        proportions = proportions / proportions.sum()  # 确保比例之和为1

        # 划分数据
        start_idx = 0
        for i in range(args.client_num_in_total):
            client_idx = i
            # 根据比例计算该客户端的数据量
            num_samples = int(proportions[i] * len(all_train_x))
            end_idx = start_idx + num_samples

            client_train_x = all_train_x[start_idx:end_idx]
            client_train_y = all_train_y[start_idx:end_idx]
            client_test_x = all_test_x[start_idx:end_idx]
            client_test_y = all_test_y[start_idx:end_idx]

            # transform to batches
            train_batch = batch_data(args, {"x": client_train_x, "y": client_train_y}, batch_size)
            test_batch = batch_data(args, {"x": client_test_x, "y": client_test_y}, batch_size)

            # index using client index
            train_data_local_dict[client_idx] = train_batch
            test_data_local_dict[client_idx] = test_batch
            train_data_local_num_dict[client_idx] = len(client_train_x)

            # print(f"Processed client {client_idx}: train data {len(client_train_x)}, test data {len(client_test_x)}")

            start_idx = end_idx

        # 处理剩余数据（由于整数转换可能导致的剩余）
        if start_idx < len(all_train_x):
            remaining_train_x = all_train_x[start_idx:]
            remaining_train_y = all_train_y[start_idx:]
            remaining_test_x = all_test_x[start_idx:]
            remaining_test_y = all_test_y[start_idx:]

            # 将剩余数据分配给最后一个客户端
            last_client_idx = args.client_num_in_total - 1
            # 检查是否已经有数据
            if last_client_idx in train_data_local_dict and train_data_local_dict[last_client_idx]:
                # 获取最后一个客户端的所有批次
                last_train_batches = train_data_local_dict[last_client_idx]
                last_test_batches = test_data_local_dict[last_client_idx]
                
                # 检查是否有批次
                if last_train_batches and last_test_batches:
                    # 将剩余数据添加到最后一个批次
                    new_train_batch_x = np.concatenate([last_train_batches[-1][0], remaining_train_x])
                    new_train_batch_y = np.concatenate([last_train_batches[-1][1], remaining_train_y])
                    new_test_batch_x = np.concatenate([last_test_batches[-1][0], remaining_test_x])
                    new_test_batch_y = np.concatenate([last_test_batches[-1][1], remaining_test_y])
                    
                    # 更新最后一个客户端的数据
                    new_train_batches = last_train_batches[:-1] + [(new_train_batch_x, new_train_batch_y)]
                    new_test_batches = last_test_batches[:-1] + [(new_test_batch_x, new_test_batch_y)]
                    train_data_local_dict[last_client_idx] = new_train_batches
                    test_data_local_dict[last_client_idx] = new_test_batches
                    train_data_local_num_dict[last_client_idx] += len(remaining_train_x)
                    # print(f"Added remaining data to client {last_client_idx}: train data {train_data_local_num_dict[last_client_idx]}")
                else:
                    # 如果没有批次，直接创建一个新批次
                    train_batch = batch_data(args, {"x": remaining_train_x, "y": remaining_train_y}, batch_size)
                    test_batch = batch_data(args, {"x": remaining_test_x, "y": remaining_test_y}, batch_size)
                    train_data_local_dict[last_client_idx] = train_batch
                    test_data_local_dict[last_client_idx] = test_batch
                    train_data_local_num_dict[last_client_idx] = len(remaining_train_x)
                    # print(f"Added remaining data to client {last_client_idx}: train data {len(remaining_train_x)}, test data {len(remaining_test_x)}")
            else:
                # 如果最后一个客户端没有数据，直接创建一个新条目
                train_batch = batch_data(args, {"x": remaining_train_x, "y": remaining_train_y}, batch_size)
                test_batch = batch_data(args, {"x": remaining_test_x, "y": remaining_test_y}, batch_size)
                train_data_local_dict[last_client_idx] = train_batch
                test_data_local_dict[last_client_idx] = test_batch
                train_data_local_num_dict[last_client_idx] = len(remaining_train_x)
                # print(f"Added remaining data to client {last_client_idx}: train data {len(remaining_train_x)}, test data {len(remaining_test_x)}")

    client_num = args.client_num_in_total
    class_num = 10

    logging.info("finished the loading data")
    print(f"Total clients: {client_idx}")
    print(f"Total train data: {train_data_num}")
    print(f"Total test data: {test_data_num}")
    print(f"Train data local num dict: {train_data_local_num_dict}")

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )