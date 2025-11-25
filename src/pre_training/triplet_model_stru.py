import os
import torch
from tensorflow import keras
from tensorflow.keras import Input, Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate
from model_base import ModelBase
# from BertConfiguration import BertConfig
# from TheBertEmbedding import BertEmbedding

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
        # self.model = self.base_model()
        self.model = self.triplet_loss_model()  # 使用Triplet Loss训练Model
        self.base_model = self.base_model()

    def triplet_loss_model(self):
        # structure_anc_input = Input(shape=(60, 180), name='structure_anc_input')  # anchor
        # structure_pos_input = Input(shape=(60, 180), name='structure_pos_input')  # positive
        # structure_neg_input = Input(shape=(60, 180), name='structure_neg_input')  # negative
        structure_anc_input = Input(shape=(50, 265), name='structure_anc_input')  # anchor
        structure_pos_input = Input(shape=(50, 265), name='structure_pos_input')  # positive
        structure_neg_input = Input(shape=(50, 265), name='structure_neg_input')  # negative

        shared_model = self.base_model()  # 共享模型

        std_out = shared_model(structure_anc_input)
        pos_out = shared_model(structure_pos_input)
        neg_out = shared_model(structure_neg_input)

        print("[INFO] model - 锚shape: %s" % str(std_out.get_shape()))
        print("[INFO] model - 正shape: %s" % str(pos_out.get_shape()))
        print("[INFO] model - 负shape: %s" % str(neg_out.get_shape()))

        output = Concatenate()([std_out, pos_out, neg_out])  # 连接

        print("[INFO] model -output:%s" % str(output.get_shape()))

        model = keras.Model(inputs=[structure_anc_input, structure_pos_input, structure_neg_input], outputs=output)

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

        margin = 0.3  # 这是你的三元组损失中的边界值
        basic_loss = distance_positive - distance_negative + margin
        loss = K.maximum(basic_loss, 0.0)

        print("[INFO] model - triplet_loss shape: %s" % str(loss.shape))
        return loss

    def base_model(self):
        """
        Triplet Loss的基础网络，可以替换其他网络结构
        """
        structure_input = keras.Input(shape=(50, 265), name='structure')
        structure_reshape = keras.layers.Reshape((50, 265, 1), name='reshape')(structure_input)
        structure_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='conv1')(
            structure_reshape)
        structure_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2, name='pool1')(structure_conv1)
        structure_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='conv2')(
            structure_pool1)
        structure_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2, name='pool2')(structure_conv2)
        structure_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', name='conv3')(
            structure_pool2)
        structure_pool3 = keras.layers.MaxPool2D(pool_size=3, strides=3, name='pool3')(structure_conv3)
        structure_flatten = keras.layers.Flatten(name='flatten')(structure_pool3)
        dense1 = keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                                    name='dense1')(
            structure_flatten)
        drop = keras.layers.Dropout(0.5, name='drop')(dense1)
        dense2 = keras.layers.Dense(units=16, activation='relu', name='random_detail')(drop)
        dense3 = keras.layers.Dense(1, activation='sigmoid', name='dense3')(dense2)
        model = keras.Model(structure_input, dense3)

        # model.save_weights('../Experimental output/t.h5')
        # print('successful')

        return model


    # def base_model(self):
    #     """
    #     Triplet Loss的基础网络，可以替换其他网络结构
    #     """
    #     # 输入张量，包含锚、正例和负例
    #     structure_anc_input = Input(shape=(50, 305), name='structure_anc_input')  # anchor
    #     structure_pos_input = Input(shape=(50, 305), name='structure_pos_input')  # positive
    #     structure_neg_input = Input(shape=(50, 305), name='structure_neg_input')  # negative
    #
    #     # 构建模型结构
    #     # 这里使用与之前相同的模型结构，只是输入改为三元组形式
    #     structure_reshape = keras.layers.Reshape((50, 305, 1), name='reshape')
    #     structure_conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='conv1')
    #     structure_pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2, name='pool1')
    #     structure_conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='conv2')
    #     structure_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2, name='pool2')
    #     structure_conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', name='conv3')
    #     structure_pool3 = keras.layers.MaxPool2D(pool_size=3, strides=3, name='pool3')
    #     structure_flatten = keras.layers.Flatten(name='flatten')
    #     dense1 = keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001),
    #                                 name='dense1')
    #     drop = keras.layers.Dropout(0.5, name='drop')
    #     dense2 = keras.layers.Dense(units=16, activation='relu', name='random_detail')
    #     dense3 = keras.layers.Dense(1, activation='sigmoid', name='dense3')
    #
    #     # 应用模型结构到每个输入
    #     structure_anc_output = dense3(dense2(drop(dense1(structure_flatten(structure_pool3(structure_conv3(
    #         structure_pool2(
    #             structure_conv2(structure_pool1(structure_conv1(structure_reshape(structure_anc_input))))))))))))
    #     structure_pos_output = dense3(dense2(drop(dense1(structure_flatten(structure_pool3(structure_conv3(
    #         structure_pool2(
    #             structure_conv2(structure_pool1(structure_conv1(structure_reshape(structure_pos_input))))))))))))
    #     structure_neg_output = dense3(dense2(drop(dense1(structure_flatten(structure_pool3(structure_conv3(
    #         structure_pool2(
    #             structure_conv2(structure_pool1(structure_conv1(structure_reshape(structure_neg_input))))))))))))
    #     output = Concatenate()([structure_anc_output, structure_pos_output, structure_neg_output])
    #
    #     # 构建模型
    #     model = keras.Model(inputs=[structure_anc_input, structure_pos_input, structure_neg_input], outputs=output)
    #     model.compile(loss=self.triplet_loss, optimizer=Adam(), metrics=['acc', 'Recall', 'Precision', 'AUC',
    #                                                                      'TruePositives', 'TrueNegatives',
    #                                                                      'FalseNegatives', 'FalsePositives'])
    #
    #     return model


