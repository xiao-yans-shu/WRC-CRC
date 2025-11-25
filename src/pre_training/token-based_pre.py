import math
import os
import numpy

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import random
import re
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel
from triplet_model import TheTripletModel
from tensorflow.python.util import tf_inspect

import tensorflow as tf
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.python.keras import regularizers
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.util import tf_inspect
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score
from BertConfiguration import BertConfig
from TheBertEmbedding import BertEmbedding

# The following part is about defining relevant data path
# structure_dir = '../Dataset/Processed_Dataset/Structure'
# texture_dir = '../Dataset/Processed_Dataset/Texture'
# texture_dir = '../Dataset/Python Code'
texture_dir = r'C:\Users\ROG\Desktop\WCL-CRC-Readability\Readability-Features-master\Dataset\pre-training\Texture\java'

# Use for texture data preprocessing
pattern = "[A-Z]"  # [A-Z] 表示一个区间，匹配所有大写字母
pattern1 = '["\\[\\]\\\\]'  # 如果需要匹配文本中的字符“\”，在正则表达式中需要4个“\”，首先，前2个“\”和后两个“\”在python解释器中分别转义成一个“\”，然后转义后的2个“\”在正则中被转义成一个“\”
pattern2 = "[*.+!$#&,;{}()':=/<>%-]"  # 匹配特殊符号
pattern3 = '[_]'  # 匹配_

# Define basic parameters
max_len = 100
training_samples = 147
validation_samples = 63
max_words = 1000

# store all data
data_set = {}  # key是文件名，value是标签0/1

# store file name
file_name = []

# store structure information
data_structure = {}

# store texture information
data_texture = {}

# store token, position and segment information
data_token = {}
data_position = {}
data_segment = {}
# dic_content = {}

# store the content of each text
string_content = {}  #

# 实验部分  --  随机打乱数据
all_data = []
train_data = []
test_data = []

structure = []
image = []
label = []
token = []
segment = []

# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", model_max_length=1200)   #分词器
# model = TFAutoModel.from_pretrained("microsoft/codebert-base")         #codebert
# print('Successfully load the codebertTokenizer')
tokenizer_path = 'C:/Users/ROG/Desktop/bert'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
print('Successfully load the BertTokenizer')




def process_texture_data():
    for label_type in ['Readable', 'Unreadable']:
        dir_name = os.path.join(texture_dir, label_type)
        for f_name in os.listdir(dir_name):  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
            if f_name[-4:] == ".txt":
                #print("ok")
                file_name.append(f_name.split('.')[0])
                list_content = []
                list_position = []
                list_segment = []
                s = ''
                segment_id = 0
                position_id = 0
                count = 0
                f = open(os.path.join(dir_name, f_name), errors='ignore')
                for content in f:  # 正则表达式前的 r 表示原生字符串（rawstring），该字符串声明了引号中的内容表示该内容的原始含义，避免了多次转义造成的反斜杠困扰。
                    content = re.sub(r"([a-z]+)([A-Z]+)", r"\1 \2", content)  # hasNextInt => has Next Int
                    content = re.sub(pattern1, lambda x: " " + x.group(0) + " ", content)  # group(0) 就是匹配正则表达式整体结果
                    content = re.sub(pattern2, lambda x: " " + x.group(0) + " ", content)
                    content = re.sub(pattern3, lambda x: " ", content)  # replace underscores with whitespaces for identifiers using the underscore case (e.g.,‘‘has_next_int’’ is converted into ‘‘has next int’’)
                    list_value = content.split()  # 返回分割后的字符串列表. 以空格为分隔符，包含 \n
                    for item in list_value:
                        if len(item) > 1 or not item.isalpha():  # 两个条件有一个成立时判断条件成功。   isalpha() 方法检测字符串是否只由字母组成。
                            s = s + ' ' + item
                            list_content.append(item)
                            if count < max_len:
                                list_position.append(position_id)
                                position_id += 1
                                list_segment.append(segment_id)
                            count += 1
                    segment_id += 1  # 每一行视为一个句子
                while count < max_len:
                    list_segment.append(segment_id)
                    list_position.append(count)
                    count += 1
                f.close()

                if label_type == 'Readable':  # 在这把所有数据的标签变成0（可读），1（不可读）
                    data_set[f_name.split('.')[
                        0]] = 0  # 因为data_set是字典，字典里的key是文件名（eg.Buse4),无论是structure还是texture文件夹里的文件名称都一样，所以这两个文件夹里的文件都被打上了一样的标签
                else:
                    data_set[f_name.split('.')[0]] = 1

                string_content[f_name.split('.')[0]] = s  # 得到对应key的value. eg.得到key='Buse4',s为对应的value.
                data_position[f_name.split('.')[0]] = list_position
                data_segment[f_name.split('.')[0]] = list_segment
                # dic_content[f_name.split('.')[0]] = list_content

        for sample in string_content:
            list_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(string_content[
                                                                                sample]))  # tokenizer.tokenize：仅进行分token操作； tokenizer.convert_tokens_to_ids：将token转化为对应的token index
            list_token = list_token[:max_len]
            while len(list_token) < max_len:
                list_token.append(0)  # 补零
            data_token[sample] = list_token  # data_token（字典）存储的是转化成token index的token信息


def create_pairs(x, digit_indices, num_classes):
    """
            创建正例和负例的Pairs
            :param x: 数据
            :param digit_indices: 不同类别的索引列表
            :param num_classes: 类别
            :return: Triplet Loss 的 Feed 数据
            """
    pairs = []
    label_pairs = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
    for d in range(num_classes):
        for i in range(n):
            l = []
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z3 = digit_indices[dn][i]
            pairs += [[x[z1], x[z2], x[z3]]]
            l.append(label[z1])
            l.append(label[z2])
            l.append(label[z3])
            label_pairs.append(l)
    return np.array(pairs), label_pairs


def random_dataSet():
    count_id = 0
    while count_id < 6340 * 2:
        index_id = random.randint(0, len(file_name) - 1)  # 返回指定范围内的整数
        all_data.append(file_name[index_id])  # file_name存储了每个文件的名字（不包括.txt）
        file_name.remove(file_name[index_id])  # 随机从文件夹中选取一个文件
        count_id += 1  # 一共210个数据
    for item in all_data:  # all_data存储的是所有文件名
        label.append(data_set[item])
        # structure.append(data_structure[item])
        token.append(data_token[item])
        segment.append(data_segment[item])  # 这三行分别将随机取出的这个文件对应的1.标签，2.token,3.segment信息存到了三个列表里，索引顺序和all_data一样


def merge1(a, b):
    li = []
    i = 0
    while i < len(a):
        l = []
        l.append(a[i])
        l.append(b[i])
        i = i + 1
        li.append(l)
    return li


def merge(a, b):
    li = []
    i = 0
    while i < len(a):
        li.append(a[i])
        li.append(b[i])
        i = i + 1
    return li


class MNIST():
    def __init__(self, label, token, segment, train=True):
        self.is_train = train

        if self.is_train:  # 加载训练集数据
            self.labels = label
            self.tokens = token
            self.segments = segment
            self.index = np.arange(len(self.tokens))  # 索引值，第几个样本
            print(self.index)
        else:  # 加载测试集数据
            self.tokens = token
            self.segments = segment

    def __len__(self):
        return len(self.tokens)

    def split(self, item):
        #item = random.randint(0, len(self.labels) - 1)
        anchor_tokens = self.tokens[item]
        anchor_segments = self.segments[item]

        if self.is_train:
            l = []
            anchor_label = self.labels[item]
            positive_list = self.index[self.index != item][
                self.labels[self.index != item] == anchor_label]  # 正 筛选出的位置索引
            positive_item = random.choice(positive_list)
            positive_tokens = self.tokens[positive_item]
            positive_segments = self.segments[positive_item]
            # positive_tokens, positive_segments = self.deleteline(positive_item)

            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_tokens = self.tokens[negative_item]
            negative_segments = self.segments[negative_item]
            l.append(anchor_label)
            l.append(anchor_label)
            l.append(self.labels[negative_item])
            return anchor_tokens, anchor_segments, positive_tokens, positive_segments, negative_tokens, negative_segments, l

        else:
            return anchor_tokens, anchor_segments


if __name__ == '__main__':
    # preprocess_structure_data()
    process_texture_data()
    random_dataSet()

    # format the data
    label = np.asarray(label)
    # structure = np.asarray(structure)
    token = np.asarray(token)
    segment = np.asarray(segment)

    # print('Shape of structure data tensor:', structure.shape)
    print('Shape of token tensor:', token.shape)
    print('Shape of segment tensor:', segment.shape)
    print('Shape of label tensor:', label.shape)
    # print(label)

    # train_structure = structure
    # train_image = image
    train_token = token
    train_segment = segment
    train_label = label

    k_fold = 10  # (改改这里）
    num_sample = math.ceil(len(train_label) / k_fold)
    train_vst_acc = []

    train_s_acc = []

    history_vst_list = []

    history_s_list = []

    random_forest_score = []
    random_forest_score_f1 = []
    random_forest_score_mcc = []
    random_forest_score_roc = []
    knn_score = []
    knn_score_f1 = []
    knn_score_roc = []
    knn_score_mcc = []
    svm_score = []

    config = None

    anc_token_ins = []
    pos_token_ins = []
    neg_token_ins = []
    anc_segment_ins = []
    pos_segment_ins = []
    neg_segment_ins = []
    triplet_label = []

    train_ds = MNIST(label, token, segment, train=True)
    for i in range(2500):
        anchor_tokens, anchor_segments, positive_tokens, positive_segments, negative_tokens, negative_segments, _label = train_ds.split(
            i)
        anc_token_ins.append(anchor_tokens)
        pos_token_ins.append(positive_tokens)
        neg_token_ins.append(negative_tokens)
        anc_segment_ins.append(anchor_segments)
        pos_segment_ins.append(positive_segments)
        neg_segment_ins.append(negative_segments)
        triplet_label.append(_label)

    anc_token_ins = np.asarray(anc_token_ins)
    pos_token_ins = np.asarray(pos_token_ins)
    neg_token_ins = np.asarray(neg_token_ins)
    anc_segment_ins = np.asarray(anc_segment_ins)
    pos_segment_ins = np.asarray(pos_segment_ins)
    neg_segment_ins = np.asarray(neg_segment_ins)
    triplet_label = np.asarray(triplet_label)

    print('Shape of anc_token_ins tensor:', anc_token_ins.shape)
    print('Shape of pos_token_ins tensor:', pos_token_ins.shape)
    print('Shape of neg_token_ins tensor:', neg_token_ins.shape)
    print('Shape of anc_segment_ins tensor:', anc_segment_ins.shape)
    print('Shape of pos_segment_ins tensor:', pos_segment_ins.shape)
    print('Shape of neg_segment_ins tensor:', neg_segment_ins.shape)
    print('Shape of triplet_label tensor:', triplet_label.shape)

    X = {
        'token_anc_input': anc_token_ins,
        'token_pos_input': pos_token_ins,
        'token_neg_input': neg_token_ins,
        'segment_anc_input': anc_segment_ins,
        'segment_pos_input': pos_segment_ins,
        'segment_neg_input': neg_segment_ins,
    }

    # anc_token_ins= np.asarray(anc_token_ins)
    # pos_token_ins=np.asarray(pos_token_ins)
    # print('Shape of anc_token_ins tensor:', anc_token_ins.shape)
    # print('Shape of pos_token_ins tensor:', pos_token_ins.shape)
    # print('Shape of neg_token_ins tensor:', neg_token_ins.shape)
    # print('Shape of anc_segment_ins tensor:', anc_segment_ins.shape)
    # print('Shape of pos_segment_ins tensor:', pos_segment_ins.shape)
    # print('Shape of neg_segment_ins tensor:', neg_segment_ins.shape)

    # 划分数据集
    x_anc_train_token, x_anc_val_token, x_anc_test_token = np.split(anc_token_ins, [int(.7 * len(anc_token_ins)),
                                                                                    int(.9 * len(anc_token_ins))])
    x_pos_train_token, x_pos_val_token, x_pos_test_token = np.split(pos_token_ins, [int(.7 * len(pos_token_ins)),
                                                                                    int(.9 * len(pos_token_ins))])
    x_neg_train_token, x_neg_val_token, x_neg_test_token = np.split(neg_token_ins, [int(.7 * len(neg_token_ins)),
                                                                                    int(.9 * len(neg_token_ins))])

    x_anc_train_segment, x_anc_val_segment, x_anc_test_segment = np.split(anc_segment_ins,
                                                                          [int(.7 * len(anc_segment_ins)),
                                                                           int(.9 * len(anc_segment_ins))])
    x_pos_train_segment, x_pos_val_segment, x_pos_test_segment = np.split(pos_segment_ins,
                                                                          [int(.7 * len(pos_segment_ins)),
                                                                           int(.9 * len(pos_segment_ins))])
    x_neg_train_segment, x_neg_val_segment, x_neg_test_segment = np.split(neg_segment_ins,
                                                                          [int(.7 * len(neg_segment_ins)),

                                                                           int(.9 * len(neg_segment_ins))])

    y_train, y_val, y_test = np.split(triplet_label, [int(.7 * len(triplet_label)), int(.9 * len(triplet_label))])

    print('VST_model:')
    VST_model = TheTripletModel(config=config).model

    filepath_vst = r"C:\Users\ROG\Desktop\WCL-CRC-Readability\Readability-Features-master\Experimental output\VST_BEST.hdf5"

    checkpoint_vst = ModelCheckpoint(filepath_vst, monitor='val_acc', verbose=1, save_best_only=True,
                                     save_weights_only=True,
                                     model='max')
    # mc = ModelCheckpoint('../Experimental output/s.h5', verbose=1, mode='min', save_best_only=True,
    # save_weights_only=True)
    callbacks_vst_list = [checkpoint_vst]

    history_vst = VST_model.fit([[x_anc_train_token, x_anc_train_segment], [x_pos_train_token, x_pos_train_segment],
                                 [x_neg_train_token, x_neg_train_segment]], y_train,
                                epochs=200, batch_size=64, callbacks=callbacks_vst_list, verbose=0,
                                validation_data=(
                                [[x_anc_val_token, x_anc_val_segment], [x_pos_val_token, x_pos_val_segment],
                                 [x_neg_val_token, x_neg_val_segment]], y_val))

    history_vst_list.append(history_vst)

    TheTripletModel(config=config).base_model.save_weights('../Experimental output/s_pyhton.h5')

    best_val_f1_vst = []
    best_val_auc_vst = []
    best_val_mcc_vst = []

    epoch_time_vst = 1
    for history_item in history_vst_list:
        MCC_vst = []
        F1_vst = []
        history_dict = history_item.history
        print(history_dict.keys())
        val_acc_values = history_dict['val_acc']
        val_recall_value = history_dict['val_recall']
        val_precision_value = history_dict['val_precision']
        val_auc_value = history_dict['val_auc']
        print('val_auc:', val_auc_value)
        val_false_negatives = history_dict['val_false_negatives']
        print('val_false_negatives:', val_false_negatives)
        val_false_positives = history_dict['val_false_positives']
        print('val_false_positives:', val_false_positives)
        val_true_positives = history_dict['val_true_positives']
        print('val_true_positives:', val_true_positives)
        val_true_negatives = history_dict['val_true_negatives']
        print('val_true_negatives:', val_true_negatives)
        for i in range(20):
            tp = val_true_positives[i]
            tn = val_true_negatives[i]
            fp = val_false_positives[i]
            fn = val_false_negatives[i]
            if tp > 0 and tn > 0 and fn > 0 and fp > 0:
                result_mcc = (tp * tn - fp * fn) / max(
                    (math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))), 1e-9)
                MCC_vst.append(result_mcc)
                result_precision = tp / max((tp + fp), 1e-9)
                result_recall = tp / (tp + fn)
                result_f1 = 2 * result_precision * result_recall / max((result_precision + result_recall), 1e-9)
                F1_vst.append(result_f1)
        train_vst_acc.append(np.max(val_acc_values))
        #        best_val_f1_vst.append(np.max(F1_vst))
        best_val_auc_vst.append(np.max(val_auc_value))
        best_val_mcc_vst.append(np.max(MCC_vst))
        print('Processing fold #', epoch_time_vst)
        print('------------------------------------------------')
        print('best accuracy score is #', np.max(val_acc_values))
        print('average recall score is #', np.mean(val_recall_value))
        print('average precision score is #', np.mean(val_precision_value))
        #       print('best f1 score is #', np.max(F1_vst))
        print('best auc score is #', np.max(val_auc_value))
        print('best mcc score is #', np.max(MCC_vst))
        print()
        print()
        epoch_time_vst = epoch_time_vst + 1

    print('Average vst model acc score', np.mean(train_vst_acc))
    print('Average vst model f1 score', np.mean(best_val_f1_vst))
    print('Average vst model auc score', np.mean(best_val_auc_vst))
    print('Average vst model mcc score', np.mean(best_val_mcc_vst))
    print()

    # model.save_weights('1-net.model') #保存模型参数
    # VST_model.save_weights('../Experimental output/pretrain_weights.h5')
