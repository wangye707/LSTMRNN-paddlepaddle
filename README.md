# LSTMRNN-paddlepaddle
基于深度学习的英文文本分类

数据集下载：
* 百度AiStudio公开数据集下载地址：https://aistudio.baidu.com/aistudio/datasetdetail/54118

数据集以及路径介绍：
```
en_text   //下载后的数据集放在这里
├── unsup //未分类的英文本文，将是本次任务的预测数据
├── test   //测试集数据集，其中包含pos和neg两类数据
    ├── pos       
    └── neg
└── train  //训练集数据集，其中包含pos和neg两类数据
    ├── pos       
    └── neg
├── train.py //训练脚本
├── infer.py //预测脚本
├── model_path //模型保存路径，执行train.py生成
    ├── 500   //默认每500步和最后一步保存一次，名称后缀不用管
    └── 1000
├── net.py //网络结构
├── save_pre.txt //执行infer.py生成的预测结果
├── word2id_dict //执行train.py生成的单词对照表

```

训练环境关键包依赖
* paddlepaddle-gpu == 1.8.4.post97

执行方式（注意，以下脚本有生成文件指令，请附带sudo权限）：
```
1.开始训练
python train.py 
说明：此处会调用readdata.py中的函数，生成字典文件‘word2id_dict’
      训练的默认超参数均在train.py中可以查看
      模型会保存至model_path路径中
      因GPU设备环境限制可以将代码中的use_gpu = True改为False，采用cpu训练
2.评估网络模型
python infer.py 
说明：infer.py中的model_path是指定的模型文件路径，例如默认的为：model_path = 'model_path/500'
      预测的文件会生成至save_path文件中，生成为txt文件
      因GPU设备环境限制可以将代码中的use_gpu = True改为False，采用cpu训练
```
训练部分迭代过程展示：
```
step 950, loss 0.002
step 960, loss 0.001
step 970, loss 0.000
the acc in the test set is 0.783
```
预测过程展示：
```
预测结果保存路径： save_pre.txt 过程较慢，耐心等待，有空点赞以及留言等，谢谢各位啦~
保存进度： 1 %
保存进度： 2 %
保存进度： 3 %
保存进度： 4 %
```
预测结果部分展示：
```
en_text/unsup/42446_0.txt------->0
en_text/unsup/3356_0.txt------->0
en_text/unsup/5675_0.txt------->0
en_text/unsup/44521_0.txt------->1
en_text/unsup/12931_0.txt------->1
en_text/unsup/719_0.txt------->0
en_text/unsup/37765_0.txt------->0
```
