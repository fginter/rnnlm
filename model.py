import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, TimeDistributed, Dropout, BatchNormalization, Add
from tensorflow.keras.layers import CuDNNLSTM as LSTM


# class RawRNNLayer(Layer):

#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(MyLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

#     def call(self, x):
#         return K.dot(x, self.kernel)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)



class LSTMModel_1:

    def __init__(self,vocab_size):
        self.model=self.make_model(vocab_size)


    def make_model(self,vocab_size):
        inp=Input(shape=(None,),dtype=tf.int64,name="input_sent")
        emb_layer=Embedding(input_dim=vocab_size,output_dim=768,embeddings_initializer=tf.keras.initializers.Constant(0.01))
        emb=emb_layer(inp)
        emb_drop=Dropout(0.2)(emb)
        emb_drop_n=BatchNormalization()(emb_drop)
        rnn1=LSTM(768,return_sequences=True)(emb_drop_n)
        rnn1_n=BatchNormalization()(rnn1)
        rnn2=LSTM(768,return_sequences=True)(rnn1_n)
        rnn2_n=BatchNormalization()(rnn2)
        proj_weights=TimeDistributed(Dense(512,activation="tanh"))(Add()([rnn1_n,rnn2_n]))
        dec=Dense(vocab_size,activation="softmax",name="decision")
        dec_td=TimeDistributed(dec)(proj_weights)
        mod=tf.keras.Model(inputs=[inp],outputs=[dec_td])
        return mod
