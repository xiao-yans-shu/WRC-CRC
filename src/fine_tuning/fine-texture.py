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
from tensorflow.python.keras import regularizers
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.util import tf_inspect
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score
#from sklearn.metrics._plot import _roc_curve
from transformers import AutoTokenizer, TFAutoModel


# Define the basic bert class
class BertConfig(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.pop('vocab_size', 30000)
        self.type_vocab_size = kwargs.pop('type_vocab_size', 300)
        self.hidden_size = kwargs.pop('hidden_size', 768)
        self.num_hidden_layers = kwargs.pop('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 12)
        self.intermediate_size = kwargs.pop('intermediate_size', 3072)
        self.hidden_activation = kwargs.pop('hidden_activation', 'gelu')
        self.hidden_dropout_rate = kwargs.pop('hidden_dropout_rate', 0.1)
        self.attention_dropout_rate = kwargs.pop('attention_dropout_rate', 0.1)
        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 200)
        self.max_sequence_length = kwargs.pop('max_sequence_length', 200)


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(name='BertEmbedding')
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.token_embedding = self.add_weight('weight', shape=[self.vocab_size, self.hidden_size],
                                               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.type_vocab_size = config.type_vocab_size

        self.position_embedding = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='position_embedding'
        )
        self.token_type_embedding = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='token_type_embedding'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)

    def build(self, input_shape):
        with tf.name_scope('bert_embeddings'):
            super().build(input_shape)

    def call(self, inputs, training=False, mode='embedding'):
        # used for masked lm
        if mode == 'linear':
            return tf.matmul(inputs, self.token_embedding, transpose_b=True)

        input_ids, token_type_ids = inputs
        input_ids = tf.cast(input_ids, dtype=tf.int32)
        position_ids = tf.range(input_ids.shape[1], dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_ids.shape.as_list(), 0)

        position_embeddings = self.position_embedding(position_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).

        Returns:
            Python dictionary.
        """
        all_args = tf_inspect.getfullargspec(self.__init__).args
        config = {
            'name': self.name,
            'trainable': self.trainable,
        }
        if hasattr(self, '_batch_input_shape'):
            config['batch_input_shape'] = self._batch_input_shape
        config['dtype'] = policy.serialize(self._dtype_policy)
        if hasattr(self, 'dynamic'):
            # Only include `dynamic` in the `config` if it is `True`
            if self.dynamic:
                config['dynamic'] = self.dynamic
            elif 'dynamic' in all_args:
                all_args.remove('dynamic')
        expected_args = config.keys()
        # Finds all arguments in the `__init__` that are not in the config:
        extra_args = [arg for arg in all_args if arg not in expected_args]
        # Check that either the only argument in the `__init__` is  `self`,
        # or that `get_config` has been overridden:
        if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
            raise NotImplementedError('Layer %s has arguments in `__init__` and '
                                      'therefore must override `get_config`.' %
                                      self.__class__.__name__)
        return config


# The following part is about defining relevant data path
structure_dir = '../Dataset/Processed Dataset/Structure'
texture_dir = '../Dataset/Processed Dataset/Texture'
picture_dir = '../Dataset/Processed Dataset/Image'

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

tokenizer_path = 'C:/Users/ROG/Desktop/bert'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
print('Successfully load the BertTokenizer')


def process_texture_data():
    for label_type in ['Readable', 'Unreadable']:
        dir_name = os.path.join(texture_dir, label_type)
        for f_name in os.listdir(dir_name):
            if f_name[-4:] == ".txt":
                file_name.append(f_name.split('.')[0])
                list_content = []
                list_position = []
                list_segment = []
                s = ''
                segment_id = 0
                position_id = 0
                count = 0
                f = open(os.path.join(dir_name, f_name), errors='ignore')
                for content in f:
                    content = re.sub(r"([a-z]+)([A-Z]+)", r"\1 \2", content)
                    content = re.sub(pattern1, lambda x: " " + x.group(0) + " ", content)
                    content = re.sub(pattern2, lambda x: " " + x.group(0) + " ", content)
                    content = re.sub(pattern3, lambda x: " ", content)
                    list_value = content.split()
                    for item in list_value:
                        if len(item) > 1 or not item.isalpha():
                            s = s + ' ' + item
                            list_content.append(item)
                            if count < max_len:
                                list_position.append(position_id)
                                position_id += 1
                                list_segment.append(segment_id)
                            count += 1
                    segment_id += 1
                while count < max_len:
                    list_segment.append(segment_id)
                    list_position.append(count)
                    count += 1
                f.close()

                if label_type == 'Readable':                             #在这把所有数据的标签变成0（可读），1（不可读）
                    data_set[f_name.split('.')[0]] = 0                   #因为data_set是字典，字典里的key是文件名（eg.Buse4),无论是structure还是texture文件夹里的文件名称都一样，所以这两个文件夹里的文件都被打上了一样的标签
                else:
                    data_set[f_name.split('.')[0]] = 1

                string_content[f_name.split('.')[0]] = s
                data_position[f_name.split('.')[0]] = list_position
                data_segment[f_name.split('.')[0]] = list_segment
                # dic_content[f_name.split('.')[0]] = list_content

        for sample in string_content:
            list_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(string_content[sample]))
            list_token = list_token[:max_len]
            while len(list_token) < max_len:
                list_token.append(0)
            data_token[sample] = list_token

def random_dataSet():
    count_id = 0
    while count_id < 210:
        index_id = random.randint(0, len(file_name) - 1)
        all_data.append(file_name[index_id])
        file_name.remove(file_name[index_id])
        count_id += 1
    for item in all_data:
        label.append(data_set[item])
        token.append(data_token[item])
        segment.append(data_segment[item])


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


def create_NetS():
    bert_config = BertConfig(max_sequence_length=max_len)
    token_input = keras.Input(shape=(max_len,), name='token')  # 表示输入的将是一批100维的向量。
    segment_input = keras.Input(shape=(max_len,), name='segment')
    texture_embedded = BertEmbedding(config=bert_config)([token_input, segment_input])

    texture_conv1 = keras.layers.Conv1D(32, 5, activation='relu', name='conv1d')(texture_embedded)
    texture_pool1 = keras.layers.MaxPool1D(3, name='max_pooling1d')(texture_conv1)
    texture_conv2 = keras.layers.Conv1D(32, 5, activation='relu', name='conv1d_1')(texture_pool1)

    texture_gru = keras.layers.Bidirectional(keras.layers.LSTM(32), name='bidirectional')(texture_conv2)
    dense1 = keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                                name='dense')(texture_gru)
    drop = keras.layers.Dropout(0.5, name='dropout_1')(dense1)
    dense2 = keras.layers.Dense(units=16, activation='relu', name='random_detail')(drop)
    dense3 = keras.layers.Dense(1, activation='sigmoid', name='dense_1')(dense2)

    model = keras.Model([token_input, segment_input], dense3)

    model.load_weights(
      'D:\\星火\\Readability-Features-master (2)1\\Readability-Features-master\\Readability-Features-master\\Experimental output\\s.h5', by_name=True, skip_mismatch=True)


    for layer in model.layers:
       layer.trainable = False

    #解冻
    model.get_layer('BertEmbedding').trainable = True
    # model.get_layer('dense').trainable = True
    model.get_layer('random_detail').trainable = True
    model.get_layer('dense_1').trainable = True

    print(model.get_layer('dense').get_weights())

    rms = keras.optimizers.RMSprop(lr=0.001)
    model.summary()
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc', 'Recall', 'Precision', 'AUC',
                                                                      'TruePositives', 'TrueNegatives',
                                                                      'FalseNegatives', 'FalsePositives'])

    return model

# def create_NetS():
#     bert_config = BertConfig(max_sequence_length=max_len)
#     token_input = keras.Input(shape=(max_len,), name='token')
#     segment_input = keras.Input(shape=(max_len,), name='segment')
#     texture_embedded = BertEmbedding(config=bert_config)([token_input, segment_input])
#     texture_conv1 = keras.layers.Conv1D(32, 5, activation='relu')(texture_embedded)
#     texture_pool1 = keras.layers.MaxPool1D(3)(texture_conv1)
#     texture_conv2 = keras.layers.Conv1D(32, 5, activation='relu')(texture_pool1)
#
#     texture_gru = keras.layers.Bidirectional(keras.layers.LSTM(32))(texture_conv2)
#     dense1 = keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(texture_gru)
#     drop = keras.layers.Dropout(0.5)(dense1)
#     dense2 = keras.layers.Dense(units=16, activation='relu', name='random_detail')(drop)
#     dense3 = keras.layers.Dense(1, activation='sigmoid')(dense2)
#     model = keras.Model([token_input, segment_input], dense3)
#     rms = keras.optimizers.RMSprop(lr=0.0015)
#     # model.summary()
#     model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc', 'Recall', 'Precision', 'AUC',
#                                                                       'TruePositives', 'TrueNegatives',
#                                                                       'FalseNegatives', 'FalsePositives'])
#     return model



if __name__ == '__main__':

    process_texture_data()
    random_dataSet()

    # format the data
    label = np.asarray(label)
    token = np.asarray(token)
    segment = np.asarray(segment)

    print('Shape of token tensor:', token.shape)
    print('Shape of segment tensor:', segment.shape)
    print('Shape of label tensor:', label.shape)

    train_token = token
    train_segment = segment
    train_label = label

    k_fold = 10
    num_sample = math.ceil(len(train_label) / k_fold)
    train_vst_acc = []
    train_v_acc = []
    train_s_acc = []
    train_t_acc = []

    history_vst_list = []
    history_v_list = []
    history_s_list = []
    history_t_list = []

    random_forest_score = []
    random_forest_score_f1 = []
    random_forest_score_mcc = []
    random_forest_score_roc = []
    knn_score = []
    knn_score_f1 = []
    knn_score_roc = []
    knn_score_mcc = []
    svm_score = []

    for epoch in range(k_fold):
        print('Now is fold {}'.format(epoch))

        x_val_token = train_token[epoch * num_sample:(epoch + 1) * num_sample]
        x_val_segment = train_segment[epoch * num_sample:(epoch + 1) * num_sample]
        y_val = train_label[epoch * num_sample:(epoch + 1) * num_sample]

        x_train_token_part_1 = train_token[:epoch * num_sample]
        x_train_token_part_2 = train_token[(epoch + 1) * num_sample:]
        x_train_token = np.concatenate([x_train_token_part_1, x_train_token_part_2], axis=0)

        x_train_segment_part_1 = train_segment[:epoch * num_sample]
        x_train_segment_part_2 = train_segment[(epoch + 1) * num_sample:]
        x_train_segment = np.concatenate([x_train_segment_part_1, x_train_segment_part_2], axis=0)

        y_train_part_1 = train_label[:epoch * num_sample]
        y_train_part_2 = train_label[(epoch + 1) * num_sample:]
        y_train = np.concatenate([y_train_part_1, y_train_part_2], axis=0)

        # model training for VST, V, S, T
        S_model = create_NetS()

        filepath_s = "../Experimental output/S_BEST.hdf5"


        checkpoint_s = ModelCheckpoint(filepath_s, monitor='val_acc', verbose=1, save_best_only=True,
                                       model='max')
        callbacks_s_list = [checkpoint_s]


        history_s = S_model.fit([x_train_token, x_train_segment], y_train,
                                epochs=20, batch_size=16, callbacks=callbacks_s_list, verbose=0,
                                validation_data=([x_val_token, x_val_segment], y_val))

        history_s_list.append(history_s)



    # data analyze
    best_val_f1_vst = []
    best_val_f1_v = []
    best_val_f1_s = []
    best_val_f1_t = []
    best_val_auc_vst = []
    best_val_auc_v = []
    best_val_auc_s = []
    best_val_auc_t = []
    best_val_mcc_vst = []
    best_val_mcc_v = []
    best_val_mcc_s = []
    best_val_mcc_t = []

    epoch_time_s = 1
    for history_item in history_s_list:
        MCC_S = []
        F1_S = []
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
                MCC_S.append(result_mcc)
                result_precision = tp / (tp + fp)
                result_recall = tp / (tp + fn)
                result_f1 = 2 * result_precision * result_recall / (result_precision + result_recall)
                F1_S.append(result_f1)
        train_s_acc.append(np.max(val_acc_values))
        if len(F1_S) > 0:
            best_val_f1_s.append(np.max(F1_S))
        else:
            best_val_f1_s.append(0)
        best_val_auc_s.append(np.max(val_auc_value))
        if len(MCC_S) > 0:
            best_val_mcc_s.append(np.max(MCC_S))
        else:
            best_val_mcc_s.append(0)
        print('Processing fold #', epoch_time_s)
        print('------------------------------------------------')
        print('best accuracy score is #', np.max(val_acc_values))
        print('average recall score is #', np.mean(val_recall_value))
        print('average precision score is #', np.mean(val_precision_value))
        # print('best f1 score is #', np.max(F1_S))
        # print('best auc score is #', np.max(val_auc_value))
        # print('best mcc score is #', np.max(MCC_S))
        print()
        print()
        epoch_time_s = epoch_time_s + 1

    print("semantics")
    print('Average S model acc score', np.mean(train_s_acc))
    print('Average S model f1 score', np.mean(best_val_f1_s))
    print('Average S model auc score', np.mean(best_val_auc_s))
    print('Average S model mcc score', np.mean(best_val_mcc_s))
    print()

