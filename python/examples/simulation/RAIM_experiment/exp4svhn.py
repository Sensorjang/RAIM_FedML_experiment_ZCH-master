from raim_torch_hierarchicalfl_step_by_step_exp import raim
from raim_rs_torch_hierarchicalfl_step_by_step_exp import raim_rs
import openpyxl
import time
import argparse
import matplotlib.pyplot as plt

excel_save_path = "D:\\"

# 绘图数据
# figure: 在不同低声誉比例(0.1,0.2,0.3,0.4,0.5)下训练的最终精度
raim_acc = []
raimrs_acc = []

# 创建一个新的Excel工作簿
wb = openpyxl.Workbook()

if __name__ == '__main__':
    # 解析命令行参数
    # parser = argparse.ArgumentParser(description="Process some integers.")
    # parser.add_argument('--dataset', type=str, help='Dataset name')
    # args = parser.parse_args()
    # datasetname = args.dataset
    datasetname = 'SVHN'

    # 创建工作表
    figxls = wb.create_sheet('Figure' + '_' + datasetname)
    figxls.append(['lowrepu_ratio', '[RAIM] ACC', '[RAIM-RS] ACC'])

    esnum = 5
    ednum = 20

    for lowrepu_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        raim_acc, raim_loss = raim(False, esnum, ednum, lowrepu_ratio)
        print("[RAIM] raim_acc:{} raim_loss:{}".format(raim_acc, raim_loss))
        raimrs_acc, raimrs_loss = raim_rs(False, esnum, ednum, lowrepu_ratio)
        print("[RAIM-RS] raimrs_acc:{} raimrs_loss:{}".format(raimrs_acc, raimrs_loss))

        raim_acc.append(raim_acc[-1])
        raimrs_acc.append(raimrs_acc[-1])

    # 输出数据到excel表格
    for i in range(len(raim_acc)):
        figxls.append([0.1 * i, raim_acc[i], raimrs_acc[i]])

    wb.save(excel_save_path + 'exp4' + '_' + datasetname + '_' + str(int(time.time()))  + '.xlsx')

    # 创建一个柱状图， x轴轴为低声誉比例（0.1,0.2,0.3,0.4,0.5），y轴为最终精度，每个x对应两个并列不重叠的纵柱
    fig1, ax1 = plt.subplots()

    bar_width = 2  # 柱状图的宽度
    index = [0.1, 0.2, 0.3, 0.4, 0.5]
    index_raim = [i - bar_width / 2 for i in index]
    index_raimrs = [i + bar_width / 2 for i in index]
    ax1.bar(index_raim, raim_acc, width=bar_width, label='RAIM', alpha=0.7)
    ax1.bar(index_raimrs, raimrs_acc, width=bar_width, label='RAIM-RS', alpha=0.7)
    ax1.set_xlabel('lowrepu_ratio')
    ax1.set_ylabel('ACC')
    ax1.set_title('RAIM and RAIM-RS ACC')

    ax1.legend()

    plt.savefig(excel_save_path + 'exp4' + '_' + datasetname + '_' + str(int(time.time()))  + '.png')

    plt.show()
    plt.close()
    time.sleep(1)
