
import os
from tensorflow import keras
from tensorflow.keras import Input, Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate
from model_base import ModelBase
from BertConfiguration import BertConfig
from TheBertEmbedding import BertEmbedding

max_len = 100

class TheTripletModel(ModelBase):
    """
    TripletLoss模型
    """

    # MARGIN = 0.33 # 超参

    def __init__(self, config):
        super(TheTripletModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = self.triplet_loss_model()  # 使用Triplet Loss训练Model
        self.base_model = self.base_model()

    def triplet_loss_model(self):
        token_anc_input = Input(shape=(max_len,), name='token_anc_input')  # anchor
        token_pos_input = Input(shape=(max_len,), name='token_pos_input')  # positive
        token_neg_input = Input(shape=(max_len,), name='token_neg_input')  # negative

        segment_anc_input = Input(shape=(max_len,), name='segment_anc_input')  # anchor
        segment_pos_input = Input(shape=(max_len,), name='segment_pos_input')  # positive
        segment_neg_input = Input(shape=(max_len,), name='segment_neg_input')  # negative

        shared_model = self.base_model()  # 共享模型

        std_out = shared_model([token_anc_input, segment_anc_input])
        pos_out = shared_model([token_pos_input, segment_pos_input])
        neg_out = shared_model([token_neg_input, segment_neg_input])

        print("[INFO] model - 锚shape: %s" % str(std_out.get_shape()))
        print("[INFO] model - 正shape: %s" % str(pos_out.get_shape()))
        print("[INFO] model - 负shape: %s" % str(neg_out.get_shape()))

        output = Concatenate()([std_out, pos_out, neg_out])  # 连接

        print("[INFO] model -output:%s" % str(output.get_shape()))

        model = keras.Model(inputs=[[token_anc_input, segment_anc_input], [token_pos_input, segment_pos_input], [token_neg_input, segment_neg_input]], outputs=output)

        #plot_model(model, to_file=os.path.join(self.config.img_dir, "triplet_loss_model.png"), show_shapes=True)  # 绘制模型图
        model.compile(loss=self.triplet_loss, optimizer=Adam(), metrics=['acc', 'Recall', 'Precision', 'AUC',
                                                                      'TruePositives', 'TrueNegatives',
                                                                      'FalseNegatives', 'FalsePositives'])

        return model

    @staticmethod

    def triplet_loss(y_true, y_pred):
        """
        Triplet Loss的损失函数
        """
        anc, pos, neg = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        # 欧式距离的平方
        distance_positive = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
        distance_negative = K.sum(K.square(anc - neg), axis=-1, keepdims=True)

        margin = 0.33  # 这是你的三元组损失中的边界值
        basic_loss = distance_positive - distance_negative + margin
        loss = K.maximum(basic_loss, 0.0)

        print("[INFO] model - triplet_loss shape: %s" % str(loss.shape))
        return loss

    def base_model(self):
        """
        Triplet Loss的基础网络，可以替换其他网络结构
        """
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
        drop = keras.layers.Dropout(0.5, name='dropout_n')(dense1)
        dense2 = keras.layers.Dense(units=16, activation='relu', name='random_detail')(drop)
        dense3 = keras.layers.Dense(1, activation='sigmoid', name='dense_1')(dense2)

        model = keras.Model([token_input, segment_input], dense3)

        # model.save_weights('../Experimental output/s_python.h5')
        #mc = ModelCheckpoint('../Experimental output/s.h5', verbose=1, mode='min', save_best_only=True,
                            # save_weights_only=True)
        print('successful')

        return model


