# Define the basic codebert class

class BertConfig(object):

    def __init__(self, **kwargs):  #接受任意个数的参数，如果是有指定key的参数,会以dict的形式存放到kwargs中
        super().__init__()
        self.vocab_size = kwargs.pop('vocab_size', 70000)  # kwargs(keyword arguments)关键字参数，kwargs.pop()用来删除关键字参数中的末尾元素。 按照vocab_size在kwargs里边，找到相应的值，如果有，返回给self.vocab_size，并且从kwargs中删除掉。如果没有vocab_size，就返回50000.
        self.type_vocab_size = kwargs.pop('type_vocab_size', 300)  #允许的最大句子位置，即最多输入的句子数量
        self.hidden_size = kwargs.pop('hidden_size', 768)    #也是嵌入向量维度（词嵌入长度)。 在BERT中，每个词会被转换成768维的向量表示。
        self.num_hidden_layers = kwargs.pop('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 12)
        self.intermediate_size = kwargs.pop('intermediate_size', 3072)
        self.hidden_activation = kwargs.pop('hidden_activation', 'gelu')
        self.hidden_dropout_rate = kwargs.pop('hidden_dropout_rate', 0.1)
        self.attention_dropout_rate = kwargs.pop('attention_dropout_rate', 0.1)
        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 200)  #允许的最大标记位置
        self.max_sequence_length = kwargs.pop('max_sequence_length', 200)