import math
import os
import random
import re
import tensorflow as tf
import cv2
import numpy
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel
from triplet_model_stru import TheTripletModel
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
#from sklearn.metrics import plot_roc_curve, roc_auc_score

# The following part is about defining relevant data path
structure_dir = r'C:\Users\ROG\Desktop\WCL-CRC-Readability\Readability-Features-master\Dataset\pre-training\Structure\java'

# Use for texture data preprocessing
pattern = "[A-Z]"
pattern1 = '["\\[\\]\\\\]'
pattern2 = "[*.+!$#&,;{}()':=/<>%-]"
pattern3 = '[_]'

# Define basic parameters
max_len = 100
training_samples = 147
validation_samples = 63
max_words = 1000

# store all data
data_set = {}

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
string_content = {}

# store picture information
data_picture = {}

# store content of each picture
data_image = []

# 实验部分  --  随机打乱数据
all_data = []
train_data = []
test_data = []

structure = []
label = []

def preprocess_structure_data():
    for label_type in ['Readable', 'Unreadable']:
        dir_name = os.path.join(structure_dir, label_type)
        for f_name in os.listdir(dir_name):
            f = open(os.path.join(dir_name, f_name), errors='ignore')
            lines = []
            if not f_name.startswith('.'):
                file_name.append(f_name.split('.')[0])
                for line in f:
                    line = line.strip(',\n')
                    info = line.split(',')
                    info_int = []
                    count = 0
                    for item in info:
                        if count < 265:
                            info_int.append(int(item))
                            count += 1
                    info_int = np.asarray(info_int)
                    lines.append(info_int)
                f.close()
                lines = np.asarray(lines)
                if label_type == 'Readable':
                    data_set[f_name.split('.')[0]] = 0
                else:
                    data_set[f_name.split('.')[0]] = 1
                data_structure[f_name.split('.')[0]] = lines

def random_dataSet():
    count_id = 0
    while count_id < 6340*2:
        index_id = random.randint(0, len(file_name) - 1)
        all_data.append(file_name[index_id])
        file_name.remove(file_name[index_id])
        count_id += 1
    for item in all_data:
        label.append(data_set[item])
        structure.append(data_structure[item])


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_1 = true_positives / (possible_positives + K.epsilon())
    return recall_1


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_1 = true_positives / (predicted_positives + K.epsilon())
    return precision_1


class MNIST():
    def __init__(self,label, structure, train=True):
        self.is_train = train

        if self.is_train:#加载训练集数据
            self.labels = label
            self.structures = structure
            self.index = np.arange(len(self.structures))#索引值，第几个样本
            print('hhhhh+')
            print(self.index)
        else:#加载测试集数据
            self.structures = structure

    def __len__(self):
        return len(self.structures)

    def split(self,item):
        item = random.randint(0, len(self.labels) - 1)
        anchor_structures = self.structures[item]

        if self.is_train:
            l=[]
            anchor_label = self.labels[item]
            positive_list = self.index[self.index != item][
                self.labels[self.index != item] == anchor_label]  # 正 筛选出的位置索引
            positive_item = random.choice(positive_list)
            positive_structures = self.structures[positive_item]

            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_structures = self.structures[negative_item]

            l.append(anchor_label)
            l.append(anchor_label)
            l.append(self.labels[negative_item])
            return anchor_structures, positive_structures, negative_structures, l

        else:
            return anchor_structures



if __name__ == '__main__':
    preprocess_structure_data()
    random_dataSet()

    # format the data
    label = np.asarray(label)
    structure = np.asarray(structure)

    print('Shape of structure data tensor:', structure.shape)
    print('Shape of label tensor:', label.shape)

    train_structure = structure
    train_label = label

    k_fold = 10
    num_sample = math.ceil(len(train_label) / k_fold)
    train_t_acc = []
    history_t_list = []
    svm_score = []

    config = None

    anc_structure_ins = []
    pos_structure_ins = []
    neg_structure_ins = []
    triplet_label = []

    train_ds = MNIST(label, structure, train=True)
    for i in range(12680):
        anchor_structure, positive_structure, negative_structure,_label = train_ds.split(i)
        anc_structure_ins.append(anchor_structure)
        pos_structure_ins.append(positive_structure)
        neg_structure_ins.append(negative_structure)
        triplet_label.append(_label)

    anc_structure_ins = np.asarray(anc_structure_ins)
    pos_structure_ins = np.asarray(pos_structure_ins)
    neg_structure_ins = np.asarray(neg_structure_ins)
    triplet_label=np.asarray(triplet_label)

    print('Shape of anc_structure_ins tensor:', anc_structure_ins.shape)
    print('Shape of pos_structure_ins tensor:', pos_structure_ins.shape)
    print('Shape of neg_structure_ins tensor:', neg_structure_ins.shape)
    print('Shape of triplet_label tensor:', triplet_label.shape)

    X = {
        'structure_anc_input': anc_structure_ins,
        'structure_pos_input': pos_structure_ins,
        'structure_neg_input': neg_structure_ins,
    }


    # x_val_structure = train_structure[epoch * num_sample:(epoch + 1) * num_sample]
    # y_val = train_label[epoch * num_sample:(epoch + 1) * num_sample]
    #
    # x_train_structure_part_1 = train_structure[:epoch * num_sample]
    # x_train_structure_part_2 = train_structure[(epoch + 1) * num_sample:]
    # x_train_structure = np.concatenate([x_train_structure_part_1, x_train_structure_part_2], axis=0)
    #
    # y_train_part_1 = train_label[:epoch * num_sample]
    # y_train_part_2 = train_label[(epoch + 1) * num_sample:]
    # y_train = np.concatenate([y_train_part_1, y_train_part_2], axis=0)
    #划分数据集
    x_anc_train_structure, x_anc_val_structure, x_anc_test_structure = np.split(anc_structure_ins, [int(.7 * len(anc_structure_ins)),
                                                                                    int(.9 * len(anc_structure_ins))])
    x_pos_train_structure, x_pos_val_structure, x_pos_test_structure = np.split(pos_structure_ins, [int(.7 * len(pos_structure_ins)),
                                                                                    int(.9 * len(pos_structure_ins))])
    x_neg_train_structure, x_neg_val_structure, x_neg_test_structure = np.split(neg_structure_ins, [int(.7 * len(neg_structure_ins)),
                                                                                    int(.9 * len(neg_structure_ins))])

    y_train, y_val, y_test = np.split(triplet_label, [int(.7 * len(triplet_label)), int(.9 * len(triplet_label))])

    # model training for VST, V, S, T
    # T_model = create_NetT()
    T_model = TheTripletModel(config=config).model
    filepath_t = "../Experimental output/T_BEST.hdf5"

    checkpoint_t = ModelCheckpoint(filepath_t, monitor='val_acc', verbose=1, save_best_only=True,
                                       model='max')
    callbacks_t_list = [checkpoint_t]

    history_t = T_model.fit([x_anc_train_structure, x_pos_train_structure, x_neg_train_structure], y_train,
                                epochs=5, batch_size=64, callbacks=callbacks_t_list, verbose=0,
                                validation_data=([x_anc_val_structure, x_pos_val_structure, x_neg_val_structure], y_val))

    history_t_list.append(history_t)

    TheTripletModel(config=config).base_model.save_weights('../Experimental output/t.h5')
    # T_model.save_weights('../Experimental output/t.h5')
    print("successful")

    # data analyze
    best_val_f1_t = []
    best_val_auc_t = []
    best_val_mcc_t = []

    epoch_time_t = 1
    for history_item in history_t_list:
        MCC_T = []
        F1_T = []
        history_dict = history_item.history
        val_acc_values = history_dict['val_acc']
        val_recall_value = history_dict['val_recall']
        val_precision_value = history_dict['val_precision']
        val_auc_value = history_dict['val_auc']
        val_false_negatives = history_dict['val_false_negatives']
        val_false_positives = history_dict['val_false_positives']
        val_true_positives = history_dict['val_true_positives']
        val_true_negatives = history_dict['val_true_negatives']
        for i in range(20):
            tp = val_true_positives[i]
            tn = val_true_negatives[i]
            fp = val_false_positives[i]
            fn = val_false_negatives[i]
            if tp > 0 and tn > 0 and fn > 0 and fp > 0:
                result_mcc = (tp * tn - fp * fn) / (math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
                MCC_T.append(result_mcc)
                result_precision = tp / (tp + fp)
                result_recall = tp / (tp + fn)
                result_f1 = 2 * result_precision * result_recall / (result_precision + result_recall)
                F1_T.append(result_f1)
        train_t_acc.append(np.max(val_acc_values))
        # best_val_f1_t.append(np.max(F1_T))
        best_val_auc_t.append(np.max(val_auc_value))
        # best_val_mcc_t.append(np.max(MCC_T))
        print('Processing fold #', epoch_time_t)
        print('------------------------------------------------')
        print('best accuracy score is #', np.max(val_acc_values))
        print('average recall score is #', np.mean(val_recall_value))
        print('average precision score is #', np.mean(val_precision_value))
        # print('best f1 score is #', np.max(F1_T))
        print('best auc score is #', np.max(val_auc_value))
        # print('best mcc score is #', np.max(MCC_T))
        print()
        print()
        epoch_time_t = epoch_time_t + 1

    print('structure')
    print('Average T model acc score', np.mean(train_t_acc))
    # print('Average T model f1 score', np.mean(best_val_f1_t))
    print('Average T model auc score', np.mean(best_val_auc_t))
    # print('Average T model mcc score', np.mean(best_val_mcc_t))
    print()

