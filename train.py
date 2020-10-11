from readdata import read_file,build_batch,read_dict
import paddle.fluid as fluid
from net import SentimentClassifier


train_path = '/home/aistudio/data/data54118/en_text/train'
test_path = '/home/aistudio/data/data54118/en_text/test'
model_path = 'model_path'
batch_size = 128
epoch_num = 5
embedding_size = 256
step = 0
learning_rate = 0.01
max_seq_len = 128
vocab_size = 252173

train_data,train_label = read_file(path=train_path,train=True)
word2id_dict = read_dict('word2id_dict')
test_data,test_label = read_file(path = test_path,train=False)

use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
with fluid.dygraph.guard(place):
    # 创建一个用于情感分类的网络实例，sentiment_classifier
    sentiment_classifier = SentimentClassifier(
        embedding_size, vocab_size, num_steps=max_seq_len)
    # 创建优化器AdamOptimizer，用于更新这个网络的参数
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list=sentiment_classifier.parameters())

    for sentences, labels in build_batch(
            word2id_dict, train_data,train_label, batch_size, epoch_num, max_seq_len):
        #print(sentences[0])
        #print(labels[0])
        sentences_var = fluid.dygraph.to_variable(sentences)
        labels_var = fluid.dygraph.to_variable(labels)
        pred, loss = sentiment_classifier(batch_size,embedding_size,sentences_var, labels_var)

        loss.backward()
        adam.minimize(loss)
        sentiment_classifier.clear_gradients()

        step += 1
        if step % 10 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))
        if step % 500 == 0:
            fluid.save_dygraph(sentiment_classifier.state_dict(), model_path + '/' +str(step))
    #最后一步保存
    fluid.save_dygraph(sentiment_classifier.state_dict(), model_path + '/' + str(step))
    sentiment_classifier.eval()
    # 这里我们需要记录模型预测结果的准确率
    # 对于二分类任务来说，准确率的计算公式为：
    # (true_positive + true_negative) /
    # (true_positive + true_negative + false_positive + false_negative)
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    for sentences, labels in build_batch(
            word2id_dict, test_data,test_label, batch_size, 1, max_seq_len):

        sentences_var = fluid.dygraph.to_variable(sentences)
        labels_var = fluid.dygraph.to_variable(labels)

        # 获取模型对当前batch的输出结果
        pred, loss = sentiment_classifier(batch_size,embedding_size,sentences_var, labels_var)
        #print(labels)
        # 把输出结果转换为numpy array的数据结构
        # 遍历这个数据结构，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        pred = pred.numpy()
        for i in range(len(pred)):
            if labels[i][0] == 1:
                if pred[i][1] > pred[i][0]:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[i][1] > pred[i][0]:
                    fp += 1
                else:
                    tn += 1

    # 输出最终评估的模型效果
    print("the acc in the test set is %.3f" % ((tp + tn) / (tp + tn + fp + fn)))

