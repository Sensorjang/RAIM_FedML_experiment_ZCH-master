from raim_torch_hierarchicalfl_step_by_step_exp import raim
from raim_rs_torch_hierarchicalfl_step_by_step_exp import raim_rs
import openpyxl
import time
import argparse
import matplotlib.pyplot as plt

excel_save_path = "D:\\"

# 绘图数据
# figure: 训练轮次的增加，模型的精度和损失的变化
raim_acc = []
raim_loss = []
raimrs_acc = []
raimrs_loss = []

# 创建一个新的Excel工作簿
wb = openpyxl.Workbook()

if __name__ == '__main__':
    # 解析命令行参数
    # parser = argparse.ArgumentParser(description="Process some integers.")
    # parser.add_argument('--dataset', type=str, help='Dataset name')
    # args = parser.parse_args()
    # datasetname = args.dataset
    datasetname = 'CIFAR10'

    # 创建工作表
    figxls = wb.create_sheet('Figure' + '_' + datasetname)
    figxls.append(['round_idx', '[RAIM] ACC', '[RAIM] LOSS', '[RAIM-RS] ACC', '[RAIM-RS] LOSS'])

    esnum = 5
    ednum = 20

    raim_acc, raim_loss = raim(False, esnum, ednum, 0.0)
    print("[RAIM] raim_acc:{} raim_loss:{}".format(raim_acc, raim_loss))
    raimrs_acc, raimrs_loss = raim_rs(False, esnum, ednum, 0.0)
    print("[RAIM-RS] raimrs_acc:{} raimrs_loss:{}".format(raimrs_acc, raimrs_loss))

    # 输出数据到excel表格
    for i in range(len(raim_acc)):
        figxls.append([i, raim_acc[i], raim_loss[i], raimrs_acc[i], raimrs_loss[i]])

    wb.save(excel_save_path + 'exp3' + '_' + datasetname + '_' + str(int(time.time()))  + '.xlsx')

    # 创建一个折线图，左侧 y 轴为精度，右侧 y 轴为损失，x 轴为训练轮次
    fig1, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # 绘制 RAIM 的 ACC 和 LOSS
    ax1.plot(raim_acc, 'b-', label='RAIM ACC')
    ax2.plot(raim_loss, 'r--', label='RAIM LOSS')

    # 绘制 RAIM-RS 的 ACC 和 LOSS
    ax1.plot(raimrs_acc, 'g-', label='RAIM-RS ACC')
    ax2.plot(raimrs_loss, 'm--', label='RAIM-RS LOSS')

    # 设置标签和标题
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Accuracy', color='b')
    ax2.set_ylabel('Loss', color='r')

    # 设置图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 保存图像
    plt.savefig(excel_save_path + 'exp3' + '_' + datasetname + '_' + str(int(time.time())) + '.png', dpi=300)

    # 显示图像
    plt.show()
    plt.close()
    time.sleep(1)
