import paddle.fluid as fluid
from net import SentimentClassifier
from readdata import build_batch_infer,read_dict,text_to_num
import os

infer_path = '/home/aistudio/data/data54118/en_text/unsup'   #预测的路径
save_path = 'save_pre.txt'
def read_file_infer(path):
    files = os.listdir(path)
    total_text = []
    total_path = []
    for file in files:

        now_path = path + "/" + file
        total_path.append(now_path)
        with open(now_path,'r') as f:
            total_text.append(f.read())

    text_num = text_to_num(total_text,train=False)
    return text_num,total_path


batch_size = 1
embedding_size = 256
step = 0
learning_rate = 0.01
max_seq_len = 128
vocab_size = 252173

infer_data,total_path = read_file_infer(infer_path)
word2id_dict = read_dict('word2id_dict')
model_path = 'model_path/500'
percent = [0,0]  #百分比显示
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
with fluid.dygraph.guard(place):
    # 创建一个用于情感分类的网络实例，sentiment_classifier
    sentiment_classifier = SentimentClassifier(
        embedding_size, vocab_size, num_steps=max_seq_len)
    model_s, _ = fluid.load_dygraph(model_path)
    sentiment_classifier.load_dict(model_s)
    sentiment_classifier.eval()
    num = 0
    f = open(save_path,'w')
    print('预测结果保存路径：',save_path,'过程较慢，耐心等待，有空点赞以及留言等，谢谢各位啦~')
    for sentences in build_batch_infer(
            word2id_dict, infer_data,batch_size, max_seq_len):

        sentences_var = fluid.dygraph.to_variable(sentences)

        # 获取模型对当前batch的输出结果
        pred= sentiment_classifier(batch_size =batch_size,embedding_size = embedding_size,
                                   input = sentences_var,label = None)

        # 把输出结果转换为numpy array的数据结构
        # 遍历这个数据结构，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        pred = pred.numpy()[0]

        if pred[0]>pred[1]:
            predcit = 0
        else:
            predcit = 1

    # 输出最终评估的模型效果
        save = str(str(total_path[num]).strip()+'------->'+ str(predcit) + '\n')
        f.write(save)
        percent.append(int((num/len(total_path))*100))
        if percent[-1] != percent[-2]:
            print('保存进度：',percent[-1],'%')
        num = num + 1
    f.close()
    print("预测结束")


