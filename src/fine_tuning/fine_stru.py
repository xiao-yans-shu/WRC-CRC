import math
import os
import random
import re
import tensorflow as tf
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.util import tf_inspect
from transformers import BertTokenizer
# from sklearn.metrics import plot_roc_curve, roc_auc_score

# The following part is about defining relevant data path
structure_dir = '../Dataset/Processed Dataset/Structure'

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
image = []
label = []
token = []
segment = []

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
                        if count < 305:
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
    while count_id < 210:
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

def create_NetT():
    structure_input = keras.Input(shape=(50, 305), name='structure')
    structure_reshape = keras.layers.Reshape((50, 305, 1), name='reshape')(structure_input)
    structure_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='conv1')(structure_reshape)
    structure_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2, name='pool1')(structure_conv1)
    structure_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='conv2')(structure_pool1)
    structure_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2, name='pool2')(structure_conv2)
    structure_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', name='conv3')(structure_pool2)
    structure_pool3 = keras.layers.MaxPool2D(pool_size=3, strides=3, name='pool3')(structure_conv3)
    structure_flatten = keras.layers.Flatten(name='flatten')(structure_pool3)
    dense1 = keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense1')(
        structure_flatten)

    drop = keras.layers.Dropout(0.5, name='drop')(dense1)
    dense2 = keras.layers.Dense(units=16, activation='relu', name='random_detail')(drop)
    dense3 = keras.layers.Dense(1, activation='sigmoid', name='dense3')(dense2)
    model = keras.Model(structure_input, dense3)

    model.load_weights(
      'D:\\星火\\Readability-Features-master米庆原版\\Readability-Features-master\\Experimental output\\t.h5', by_name=True, skip_mismatch=True)
    # model.load_weights(
    #     'C:\\Users\\ROG\\Desktop\\t.h5',
    #     by_name=True, skip_mismatch=True)

    for layer in model.layers:
       layer.trainable = False

    print(model.get_layer('conv1').get_weights())

    # #解冻
    model.get_layer('reshape').trainable = True
    # model.get_layer('conv1').trainable = True
    # model.get_layer('pool1').trainable = True
    # model.get_layer('conv2').trainable = True
    # model.get_layer('pool2').trainable = True
    # model.get_layer('conv3').trainable = True
    # model.get_layer('pool3').trainable = True
    # model.get_layer('flatten').trainable = True
    model.get_layer('dense1').trainable = True
    model.get_layer('drop').trainable = True
    model.get_layer('random_detail').trainable = True
    model.get_layer('dense3').trainable = True

    rms = keras.optimizers.RMSprop(lr=0.001)
    model.summary()
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc', 'Recall', 'Precision', 'AUC',
                                                                      'TruePositives', 'TrueNegatives',
                                                                      'FalseNegatives', 'FalsePositives'])
    return model

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

    for epoch in range(k_fold):
        print('Now is fold {}'.format(epoch))
        x_val_structure = train_structure[epoch * num_sample:(epoch + 1) * num_sample]
        y_val = train_label[epoch * num_sample:(epoch + 1) * num_sample]

        x_train_structure_part_1 = train_structure[:epoch * num_sample]
        x_train_structure_part_2 = train_structure[(epoch + 1) * num_sample:]
        x_train_structure = np.concatenate([x_train_structure_part_1, x_train_structure_part_2], axis=0)

        y_train_part_1 = train_label[:epoch * num_sample]
        y_train_part_2 = train_label[(epoch + 1) * num_sample:]
        y_train = np.concatenate([y_train_part_1, y_train_part_2], axis=0)

        # model training for VST, V, S, T
        T_model = create_NetT()

        filepath_t = "../Experimental output/T_BEST.hdf5"

        checkpoint_t = ModelCheckpoint(filepath_t, monitor='val_acc', verbose=1, save_best_only=True,
                                       model='max')
        callbacks_t_list = [checkpoint_t]

        history_t = T_model.fit(x_train_structure, y_train,
                                epochs=20, batch_size=16, callbacks=callbacks_t_list, verbose=0,
                                validation_data=(x_val_structure, y_val))

        history_t_list.append(history_t)

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
        if len(F1_T) > 0:
            best_val_f1_t.append(np.max(F1_T))
        else:
            # 处理空数组的情况，比如添加默认值或者采取其他适当的操作
            best_val_f1_t.append(0) # 这里假设添加了默认值!
        #best_val_f1_t.append(np.max(F1_T))
        best_val_auc_t.append(np.max(val_auc_value))
        if len(MCC_T) > 0:
            best_val_mcc_t.append(np.max(MCC_T))
        else:
            # 处理空数组的情况，比如添加默认值或者采取其他适当的操作
            best_val_mcc_t.append(0)# 这里假设添加了默认值!

        print('Processing fold #', epoch_time_t)
        print('------------------------------------------------')
        print('best accuracy score is #', np.max(val_acc_values))
        print('average recall score is #', np.mean(val_recall_value))
        print('average precision score is #', np.mean(val_precision_value))
        print('best f1 score is #', np.max(F1_T))
        print('best auc score is #', np.max(val_auc_value))
        print('best mcc score is #', np.max(MCC_T))
        print()
        print()
        epoch_time_t = epoch_time_t + 1

    print('structure')
    print('Average T model acc score', np.mean(train_t_acc))
    print('Average T model f1 score', np.mean(best_val_f1_t))
    print('Average T model auc score', np.mean(best_val_auc_t))
    print('Average T model mcc score', np.mean(best_val_mcc_t))
    print()

    # T_model.save_weights('../Experimental output/t.h5')

