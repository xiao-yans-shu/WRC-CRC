import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.util import tf_inspect


#在tensorflow 2.x中自定义网络层最好的方法就是继承tf.keras.layers.Layer类，并且根据实际情况重写如下几个类方法
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

class BertEmbedding(tf.keras.layers.Layer):  #class 子类（父类）
# Layer 是一个可调用对象，它以一个或多个张量作为输入并输出一个或多个张量。
# _init_: 初始化类，你可以在此配置一些网络层需要的参数，并且也可以在此实例化tf.keras提供的一些基础算子比如DepthwiseConv2D，方便在call方法中应用；
    def __init__(self, config, **kwargs):   #定义自定义层属性，并使用 add_weight() 创建不依赖于输入形状的层状态变量。
                                            #这里的config对应着BertConfig类
        super().__init__(name='BertEmbedding')
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.token_embedding = self.add_weight('weight', shape=[self.vocab_size, self.hidden_size],               #initializer: 初始化器基类：所有初始化器继承这个类。
                                               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))    #TruncatedNormal:按照截尾正态分布生成随机张量的初始化器; stddev:一个Python标量或者一个标量张量。要生成的随机值的标准差。
        self.type_vocab_size = config.type_vocab_size

        self.position_embedding = tf.keras.layers.Embedding(     #Position Embeddings layer 实际上就是一个大小为 (512, 768) 的lookup表，表的第一行是代表第一个序列的第一个位置，第二行代表序列的第二个位置
            config.max_position_embeddings,             #词汇表大小
            config.hidden_size,                         #词向量的维度（768）
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),    #矩阵的初始化方法
            name='position_embedding'
        )
        self.token_type_embedding = tf.keras.layers.Embedding(     #Segment Embedding
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='token_type_embedding'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')   #层标准化层的做法是根据样本的特征数做归一化。在每一个时刻，层标准化层对每一个样本都分别计算所有特征的均值和标准差。
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)     #将Dropout应用于输入。 Dropout 包括在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。
                                                                               #rate参数(config.hidden_dropout_rate): 在 0 和 1 之间浮动。需要丢弃的输入比例。

    def build(self, input_shape):   #build() method：为BertEmbedding创建可训练权重的函数
        with tf.name_scope('bert_embeddings'):  #tf.name_scope：用于定义Python op的上下文管理器。 此上下文管理器将推送名称范围，这将使在其中添加的所有操作的名称带有前缀。
            super().build(input_shape)

    def __call__(self, inputs, training=False, mode='embedding'):   # call() method定义computation。返回该Layer的结果
        # used for masked language model
        if mode == 'linear':
            return tf.matmul(inputs, self.token_embedding, transpose_b=True)    #矩阵inputs乘以矩阵self.token_embedding, transpose_b=True 表示self.token_embedding在乘法之前转置。

        input_ids, token_type_ids = inputs   #将元组inputs的元素赋给变量input_ids和token_type_ids
        input_ids = tf.cast(input_ids, dtype=tf.int32)      #将input_ids的数据格式转化成dtype数据类型（32位int类型变量）
        position_ids = tf.range(input_ids.shape[1], dtype=tf.int32)[tf.newaxis, :]      #tf.range: 用来生成一个一维的序列张量，序列数的范围为(start,limit),增量为delta
        #一整行的意思为：生成一个shape为（j,1)的张量                                                 #tf.newaxis: 用来做张量维数扩展的
                                                                                        #shape[1]输出input_ids的列数（j）。
        if token_type_ids is None:     #判断token_type_ids是否有定义。若已被定义，则判断不成立
            token_type_ids = tf.fill(input_ids.shape.as_list(), 0)     #创建一个维度为dims（input_ids.shape.as_list()），值为value（0）的tensor对象

        position_embeddings = self.position_embedding(position_ids)           #position embedding
        token_type_embeddings = self.token_type_embedding(token_type_ids)     #segment embedding
        token_embeddings = tf.gather(self.token_embedding, input_ids)         #token embedding

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
        all_args = tf_inspect.getfullargspec(self.__init__).args     #获取函数参数的名称和默认值，返回一个命名的元组
        config = {                              #创建字典
            'name': self.name,
            'trainable': self.trainable,
        }
        if hasattr(self, '_batch_input_shape'):      #hasattr() 函数用于判断对象是否包含对应的属性。如果对象有该属性，返回True,否则返回False.
            config['batch_input_shape'] = self._batch_input_shape    #修改字典
        config['dtype'] = policy.serialize(self._dtype_policy)
        if hasattr(self, 'dynamic'):
            # Only include `dynamic` in the `config` if it is `True`
            if self.dynamic:
                config['dynamic'] = self.dynamic
            elif 'dynamic' in all_args:
                all_args.remove('dynamic')
        expected_args = config.keys()
        # Finds all arguments in the `__init__` that are not in the config:
        extra_args = [arg for arg in all_args if arg not in expected_args]       #for...[if]...语句一种简洁的构建List的方法，从for给定的List中选择出满足if条件的元素组成新的List
        # Check that either the only argument in the `__init__` is  `self`,
        # or that `get_config` has been overridden:
        if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
            raise NotImplementedError('Layer %s has arguments in `__init__` and '
                                      'therefore must override `get_config`.' %
                                      self.__class__.__name__)
        return config