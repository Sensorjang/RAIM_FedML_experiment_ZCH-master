<div align="center">
<h1 align="center">RAIM</h1>
RAIM_FedML_experiment_ZCH

![GitHub](https://img.shields.io/github/license/Sensorjang/RAIM_FedML_experiment_ZCH)
![Language](https://img.shields.io/badge/Language-Python-blue)

</div>

## 项目维护与代码编辑
- 左翠华Cuihua Zuo / 
- 祁盼Pan Qi / Sensorjang@foxmail.com

## 基于FedML框架搭建的RAIM算法验证实验
本实验修改了[FedML框架](FedML_README.md)的部分算法逻辑和环节，并且加入了一些新的算子和组件以实现RAIM的对比实验的细节<br/>
This experiment modified some algorithm logic and links of the [FedML framework](FedML_README.md), and added some new operators and components to achieve the details of the RAIM comparative experiment<br/>

## 实验涉及主要算子和组件信息
- 涉及的算法：RAIM (ours), QAIM, MaxQ
- 涉及的数据集：MNIST, FEMNIST, CIFAR10, SVHN
- 涉及的模型：CNN
<br/><br/>
- Algorithms involved: RAIM (ours), QAIM, MaxQ
- Datasets involved: MNIST, FEMNIST, CIFAR10, SVHN
- Models involved: CNN
<br/><br/>
- Datasets Download URL:<br/>
  - MNIST: https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip
  - FEMNIST: https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2
  - CIFAR10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  - SVHN: http://ufldl.stanford.edu/housenumbers/train_32x32.mat and http://ufldl.stanford.edu/housenumbers/test_32x32.mat

## 主要路径
- 实验运行路径：[RAIM_experiment](python/examples/simulation/RAIM_experiment)
- 算法API路径：[algorithm](python/fedml/simulation/sp)
- 数据集逻辑路径：[dataset](python/fedml/data)
<br/><br/>
- Experimental run path: [RAIM_experience](Python/examples/simulation/RAIM_experience)
- Algorithm API path: [algorithm](Python/fedml/simulation/sp)
- Dataset logical path: [dataset](Python/fedml/data)

## 快速开始
在['python/'](python/)目录下执行：<br/>
Execute in the ['Python/'](Python/) directory:<br/>
```bash
pip install .
```
然后执行[RAIM_experiment](python/examples/simulation/RAIM_experiment)下的各个python文件即可：<br/>
Then execute the various Python files under [RAIM_experiment](python/examples/simulation/RAIM_experiment) to:<br/>
```bash
examples:
python exp1.py --cf selected_svhn.yaml
python exp2.py --cf selected_svhn.yaml
python exp3.py --cf selected_svhn.yaml
python exp4.py --cf selected_svhn.yaml
python raim_torch_hierarchicalfl_step_by_step_exp.py --cf selected_svhn.yaml
python raim_rs_torch_hierarchicalfl_step_by_step_exp.py --cf selected_svhn.yaml
```
程序会自动绘制图像，并且在控制台输出绘图坐标信息<br/>
The program will automatically draw images and output drawing coordinate information on the console<br/>
绘图借助plt，在图像绘制中途程序处于阻塞状态，预定的全部FL轮次完成后图像会保持输出在用户界面上供保存和查看细节<br/>
Drawing with the help of PLT, the program is in a blocked state during the process of image drawing. After all predetermined FL rounds are completed, the image will remain output on the user interface for saving and viewing details<br/>

## License
该项目基于[Apache-2.0 License](LICENSE)许可证开源<br/>
This project is released under the [Apache-2.0 License](LICENSE).<br/>