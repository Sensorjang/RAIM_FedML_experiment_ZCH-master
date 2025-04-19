import fedml
import random
import math
import numpy as np
from scipy.optimize import minimize
from fedml import FedMLRunner
from scipy.sparse import csr_matrix

firstprice = 0
M = 0
N = 0
clusters = []
datasize = []
sorted_edge_indices = []
sorted_device_indices = []
theta = []
delta = []
gama = 1.0
es_rewards = []
param_a = 2.3
es_utilities = []
participation_ratios_rs = []

def calculate_utilities(initial_rewards, training_costs, reputations):
    """
    根据公式38计算每个终端设备EDi连接到边缘服务器ESj时的个体效用Uij
    返回一个效用矩阵U，其中U[i][j]表示EDi连接到ESj时的效用
    """
    global M, N, clusters, sorted_device_indices, sorted_edge_indices

    # 初始化效用矩阵
    utilities = np.zeros((N, M))

    # 当前聚类的终端设备数量
    for i in sorted_device_indices:
        for j in sorted_edge_indices:
            # 公式38
            # 计算clusters[j]中每个设备的training_costs[i] / ( reputations[i] + 1e-6 )总和 sumvalue
            sumvalue = 0 + sum((training_costs[i] / ( reputations[i] + 1e-6 )) for i in clusters[j])
            utilities[i][j] = initial_rewards[j] * (( 1 - (( len(clusters[j]) - 1 ) * training_costs[i] / ( reputations[i] + 1e-6 )) / (sumvalue + 1e-6)) ** 2 )
    
    # print("个体效用:" , str(utilities))
    return utilities


def calculate_participation_ratios(initial_rewards, clusters, training_costs, reputations):
    """
    根据公式19计算每个终端设备EDi连接到ESj时的参与比率Aij
    返回一个比率矩阵A，其中A[i][j]表示EDi连接到ESj时的比率
    """
    global M, N, datasize

    # 初始化比率矩阵
    ratios = np.zeros((N, M))
        
    for j in sorted_edge_indices:
        if not clusters[j]:
            continue  # 如果聚类为空，跳过
        for i in clusters[j]:
            # 公式19
            # 计算clusters[j]中每个设备的training_costs[i] / ( reputations[i] + 1e-6 )总和 sumvalue
            sumvalue = 0 + sum(training_costs[i] / ( reputations[i] + 1e-6 ) for i in clusters[j])
            ratios[i][j] = ((len(clusters[j]) - 1 ) * initial_rewards[j] / ((datasize[i] * reputations[i] * (sumvalue ** 2)) + 1e-6)) * (sumvalue - (len(clusters[j]) - 1 ) * (training_costs[i] / ( reputations[i] + 1e-6 )))
    
    return ratios


def assign_edge_servers(initial_rewards, training_costs, reputations):
    """
    输入:
        initial_rewards: 边缘服务器初始奖励集合
        training_costs: 终端设备单元训练成本集合
        reputations: 终端设备声誉值集合
        
    输出:
        assigned_servers: 每个终端设备对应的边缘服务器选择集合
        participation_ratios: 每个终端设备EDi连接到ESj时的参与比率Aij
    """
    global clusters, M, N, sorted_device_indices, sorted_edge_indices

    # 步骤1: 按照边缘服务器初始奖励大小排序
    sorted_edge_indices = sorted(range(M), key=lambda x: initial_rewards[x], reverse=True)
    
    # 步骤2: 按照终端设备单元训练成本与声誉值的比值排序
    sorted_device_indices = sorted(
        range(N),
        key=lambda x: training_costs[x] / (reputations[x] + 1e-6),  # 防止除零
        reverse=False
    )
    print("终端设备排序：{}".format(sorted_device_indices))
    print("边缘服务器排序：{}".format(sorted_edge_indices))
    
    # 步骤3: 初始化聚类
    clusters = [[] for _ in range(M)]
    
    # 随机分配逻辑
    assigned_servers = [0] * N  # 初始化分配结果
    
    for i in range(N):
        device_idx = sorted_device_indices[i]
        
        # 随机选择一个边缘服务器
        available_edges = [edge_idx for edge_idx in sorted_edge_indices]
        
        if not available_edges:
            # 如果所有聚类都为空，随机选择一个边缘服务器
            available_edges = sorted_edge_indices
            
        selected_edge = random.choice(available_edges)
        
        # 分配设备到随机选择的边缘服务器
        clusters[selected_edge].append(device_idx)
        assigned_servers[device_idx] = selected_edge
        print("将设备 {} 随机分配到边缘服务器 {};".format(device_idx, selected_edge), end='')
    
    # 计算效用矩阵
    ed_utilities = calculate_utilities(initial_rewards, training_costs, reputations)
    
    # 计算每个设备的总效用
    eds_total_utility = 0
    for i in range(N):
        value = ed_utilities[i][assigned_servers[i]]
        if value < 0.0: # 效用限制
            value = 0.0
        elif value > 5:
            value = 5
        eds_total_utility += value
    print("EDs总效用为：{}".format(eds_total_utility))

    # 步骤15-19: 计算每个设备的比率Aij
    participation_ratios = calculate_participation_ratios(initial_rewards, clusters, training_costs, reputations)
    
    return assigned_servers, participation_ratios, eds_total_utility


def generate_random_rewards(firstprice, M, fluctuate):
    """
    生成随机的初始奖励集合，波动
    """
    # 中间奖励值
    average_reward = firstprice

    # 生成随机波动的奖励值
    rewards = [random.uniform(average_reward * (1 - fluctuate), average_reward * (1 + fluctuate)) for _ in range(M)]

    # # 计算生成的奖励值总和
    # sum_rewards = sum(rewards)

    # # 计算总和与 target_quotation 的差值
    # difference = firstprice - sum_rewards

    # # 将差值平均分配到每个奖励值上
    # adjusted_rewards = [reward + (difference / M) for reward in rewards]

    return rewards

def calculate_es_rewards(firstprice, training_costs, reputations):
    """
    计算最优的ES的奖励
    """
    global sorted_edge_indices , delta , theta, es_rewards, clusters, param_a

    for j in sorted_edge_indices:
        if clusters[j] == []:
            es_rewards[j] = 0
        else:
            sumvalue = 0 + sum(training_costs[i] / ( reputations[i] + 1e-6 ) for i in clusters[j])
            b = ( len(clusters[j]) - 1 ) / (sumvalue + 1e-6)
            es_rewards[j] = (theta[j] / math.log(param_a)) - (delta[j] / firstprice * b)

    print("ES最优奖励:" , es_rewards)

def calculate_es_utilities(reputations, commun_costs):
    """
    计算ES的效用
    """
    global clusters, M, N, sorted_edge_indices, es_utilities, param_a, firstprice, participation_ratios_rs ,datasize ,es_rewards, delta , theta

    es_utilities = [0] * M # 初始化ES效用矩阵
    data_offset_ratio = 0.85
    
    for j in sorted_edge_indices:
        if clusters[j] == []:
            es_utilities[j] = 0
        else:
            sumvalue = sum(max(0, (participation_ratios_rs[i][j] * datasize[i] * reputations[i])) for i in clusters[j])
            # print("########DEBUG_INFO###### A:{} B:{} C:{}".format((math.log(firstprice * sumvalue + delta[j]) / math.log(param_a)), es_rewards[j] / (theta[j]+ 1e-6), commun_costs[j] * len(clusters[j])))
            es_utilities[j] = ((math.log(firstprice * sumvalue + delta[j]) / math.log(param_a)) - es_rewards[j] / (theta[j]+ 1e-6) - commun_costs[j] * len(clusters[j]))
            es_utilities[j] = 0 if es_utilities[j] < -10e5 else data_offset_ratio * es_utilities[j]
    print("ES效用:" , es_utilities)

def calculate_cs_price(training_costs, reputations):
    """
    计算CS的最优报价
    """
    global delta, gama, param_a, sorted_edge_indices, theta, clusters

    sumdelta = sum(delta)
    sumthetab = sum((theta[j] * ( len(clusters[j]) - 1 ) / ((sum( (training_costs[i] / (reputations[i] + 1e-6)) for i in clusters[j])) + 1e-6)) for j in sorted_edge_indices if clusters[j])
    price = (sumdelta + math.sqrt(sumdelta ** 2 + 4 * gama * sumdelta + 4 * gama * math.log(param_a) * sumdelta / sumthetab)) / ((2 * sumthetab / math.log(param_a)) + 2)

    print("CS最优报价:" , price)
    return price
def calculate_cs_utilities(finalprice):
    """
    计算CS效用
    """
    global gama 
    total_datasize = 0
    data_offset_ratio = 0.85
    for j in sorted_edge_indices:
        total_datasize += sum(datasize[i] for i in clusters[j])
    cs_utilities = gama * math.log(total_datasize * data_offset_ratio + 1) - finalprice * total_datasize * 1e-6
    # print("########DEBUG_INFO###### A:{} B:{} C:{}".format(gama * math.log(total_datasize + 1), finalprice * total_datasize, cs_utilities))
    
    print("CS效用:" , cs_utilities)
    return cs_utilities

def convert_to_str(cluster, sorted_edge_indices):
    # 构建字典
    custom_group_dict = {}
    for edge_idx, cluster in zip(sorted_edge_indices, clusters):
        custom_group_dict[edge_idx] = cluster # 包括空的聚类

    # 转换为字符串
    custom_group_str = "{"
    for edge_idx, cluster in custom_group_dict.items():
        custom_group_str += f"{edge_idx}: {cluster}, "
    custom_group_str = custom_group_str.rstrip(", ") + "}"

    print("custom_group_str:", custom_group_str)
    return custom_group_str

def generate_reputations(N, lowrepu_ratio):
    # 计算低声誉的数量
    low_repu_count = int(N * lowrepu_ratio)
    high_repu_count = N - low_repu_count

    # 生成低声誉的值
    low_reputations = [max(random.uniform(0, 0.5), 0.00001) for _ in range(low_repu_count)]
    # 生成高声誉的值
    high_reputations = [min(random.uniform(0.5, 1), 0.99999) for _ in range(high_repu_count)]

    # 合并低声誉和高声誉的值
    reputations = low_reputations + high_reputations

    # 打乱顺序
    random.shuffle(reputations)

    return reputations
def raim_rs(justsimulate, esnum = 0, ednum = 0, lowrepu_ratio = 0.0):
    global firstprice, M, N, clusters, datasize, sorted_edge_indices, sorted_device_indices, theta, delta, gama, es_rewards, param_a, es_utilities, participation_ratios_rs

    # init FedML framework
    args = fedml.init()

    if esnum != 0:
        M = esnum
    else:
        M = args.group_num # ES 边缘服务器数边

    if ednum != 0:
        N = ednum
    else:
        N = args.client_num_per_round # ED 缘设备数
    
    reputations = generate_reputations(N, lowrepu_ratio)
    args.reputations = reputations # 声誉值写入参数中

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim, train_data_local_num_dict = fedml.data.load(args)
    print("数据量分布：{}".format(train_data_local_num_dict))

    ############RAIM############
    firstprice = 2 #初始报价
    es_rewards = generate_random_rewards(firstprice, M , 0.00005) # ES 初始奖励集合
    training_costs = [random.uniform(0, 1) for _ in range(N)]
    # datasize = [random.randint(0, 10000) for _ in range(N)] # ED 训练数据集大小
    datasize = [train_data_local_num_dict[i] for i in range(N)]
    commun_costs = [random.uniform(0, 0.001) for _ in range(M)] # ES 单位协调和计算成本
    print("初始奖励:{}".format(es_rewards))
    print("终端设备训练成本:{}".format(training_costs))
    print("终端设备声誉值:{}".format(reputations))
    print("终端设备数据集大小:{}".format(datasize))

    assigned_servers_rs, participation_ratios_rs, eds_total_utility = assign_edge_servers(es_rewards, training_costs, reputations)
    print("终端设备分配:{}".format(assigned_servers_rs))
    print("终端设备参与比率:{}".format(csr_matrix(participation_ratios_rs)))

    theta = [0.1] * M #风险厌恶系数
    delta = [0.9] * M #奖励缩放系数

    calculate_es_rewards(firstprice, training_costs, reputations)

    calculate_es_utilities(reputations, commun_costs)

    finalprice = calculate_cs_price(training_costs, reputations)

    cs_utilities = calculate_cs_utilities(finalprice)

    es_utilities = [max(min(i, 100), -10) for i in es_utilities] # 效用限制
    # cs_utilities = max(min(cs_utilities, 1000), -5)
    social_utility = cs_utilities + sum(es_utilities) + eds_total_utility
    print("ES总效用:{}".format(sum(es_utilities)))
    print("社会效用:{}".format(social_utility))

    # 按照计算出的终端设备分配方式分组
    args.group_method = "custom" # raim强制使用自定义分组
    args.custom_group_str = convert_to_str(clusters, sorted_edge_indices)

    if justsimulate:
        return social_utility, cs_utilities
    ############RAIM############
    
    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    acc_list, loss_list = fedml_runner.run()
    return acc_list, loss_list

if __name__ == "__main__":
    raim_rs(False)