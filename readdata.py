import os
import random
import numpy as np
def read_file(path,train):
    neg_path = path + '/' + "neg"
    pos_path = path + '/' + "pos"
    neg_file = os.walk(neg_path)
    pos_file = os.walk(pos_path)
    neg_text = []
    pos_text = []
    for root,dir,file_name in neg_file:
        for name in file_name:
            now_path = root + "/" + name
            with open(now_path,'r') as f:
                neg_text.append(f.read())
    for root,dir,file_name in pos_file:
        for name in file_name:
            now_path = root + "/" + name
            with open(now_path,'r') as f:
                pos_text.append(f.read())
    total_text = neg_text + pos_text
    text_num = text_to_num(total_text,train=train)
    label = []
    for i in range(0,len(neg_text)):
        label.append(0)
    for i in range(0, len(pos_text)):
        label.append(1)

    return text_num,label




def data_lower(text):
    data_set = []
    for sentence in text:
        # 这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        # 一般来说这样的做法有助于效果提升
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")

        data_set.append(sentence)

    return data_set


def text_to_num(text_list,train):
    text_list = data_lower(text_list)
    save_path = 'word2id_dict'
    data_set = []
    if train:
        word2id_freq, word2id_dict = build_dict(text_list)
        save_path = 'word2id_dict'
        save_dict(word2id_dict,save_path)
    else:
        word2id_dict = read_dict(save_path)

    for sentence in text_list:
            # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
            # 这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
            # 如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict \
                            else word2id_dict['[oov]'] for word in sentence]
        data_set.append(sentence)

    return data_set


def build_dict(corpus):
    word_freq_dict = dict()
    for sentence in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    word2id_dict = dict()
    word2id_freq = dict()

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict


def save_dict(dict,path):
    f = open(path, 'w')
    f.write(str(dict))
    f.close()

def read_dict(path):
    f = open(path, 'r')
    dict_ = eval(f.read())
    f.close()
    return dict_


def build_batch(word2id_dict, corpus,label, batch_size, epoch_num, max_seq_len, shuffle=True):
    # 模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，
    #    每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num):

        # 每个epcoh前都shuffle一下数据，有助于提高模型训练的效果
        # 但是对于预测任务，不要做数据shuffle
        if shuffle:
            seed = 5
            random.seed(seed)
            random.shuffle(corpus)
            random.seed(seed)
            random.shuffle(label)
        num = 0
        for sentence in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([label[num]])
            num = num  + 1

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []
                
def build_batch_infer(word2id_dict, corpus,batch_size,max_seq_len):

    sentence_batch = []

    for sentence in corpus:
        sentence_sample = sentence[:min(max_seq_len, len(sentence))]
        if len(sentence_sample) < max_seq_len:
            for _ in range(max_seq_len - len(sentence_sample)):
                sentence_sample.append(word2id_dict['[pad]'])

        sentence_sample = [[word_id] for word_id in sentence_sample]

        sentence_batch.append(sentence_sample)


        if len(sentence_batch) == batch_size:
            yield np.array(sentence_batch).astype("int64")
            sentence_batch = []
