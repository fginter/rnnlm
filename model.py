import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, TimeDistributed, Dropout, BatchNormalization, Add, SpatialDropout1D, Lambda
from tensorflow.keras.layers import CuDNNLSTM as LSTM
import pickle
import json
import sys

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


class LSTMModel_2:

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
        rnn3=LSTM(768,return_sequences=True)(rnn2_n)
        rnn3_n=BatchNormalization()(rnn3)
        proj_weights=TimeDistributed(Dense(512,activation="tanh"))(Add()([rnn1_n,rnn2_n,rnn3_n]))
        dec=Dense(vocab_size,activation="softmax",name="decision")
        dec_td=TimeDistributed(dec)(proj_weights)
        mod=tf.keras.Model(inputs=[inp],outputs=[dec_td])
        return mod


class LSTMModel_3:

    def __init__(self,vocab_size):
        self.model=self.make_model(vocab_size)

    def make_model(self,vocab_size):
        inp=Input(shape=(None,),dtype=tf.int64,name="input_sent")
        emb_layer=Embedding(input_dim=vocab_size,output_dim=768,embeddings_initializer=tf.keras.initializers.Constant(0.01))
        emb=emb_layer(inp)
        emb_do=SpatialDropout1D(0.3)(emb)
        emb_drop_n=BatchNormalization()(emb_do)
        rnn1=LSTM(768,return_sequences=True)(emb_drop_n)
        rnn1_n=BatchNormalization()(rnn1)
        rnn2=LSTM(768,return_sequences=True)(rnn1_n)
        rnn2_n=BatchNormalization()(rnn2)
        rnn3=LSTM(768,return_sequences=True)(rnn2_n)
        rnn3_n=BatchNormalization()(rnn3)
        proj_weights=TimeDistributed(Dense(512,activation="tanh"))(Add()([rnn1_n,rnn2_n,rnn3_n]))
        dec=Dense(vocab_size,activation="softmax",name="decision")
        dec_td=TimeDistributed(dec)(proj_weights)
        mod=tf.keras.Model(inputs=[inp],outputs=[dec_td])

        return mod


class LSTMModel_4:

    def __init__(self,vocab_size):
        self.model=self.make_model(vocab_size)

    def make_model(self,vocab_size):
        inp=Input(shape=(None,),dtype=tf.int64,name="input_sent")
        emb_layer=Embedding(input_dim=vocab_size,output_dim=1500,embeddings_initializer=tf.keras.initializers.Constant(0.01))
        emb=emb_layer(inp)
        emb_do=SpatialDropout1D(0.5)(emb)
        emb_drop_n=BatchNormalization()(emb_do)
        rnn1=LSTM(1500,return_sequences=True)(emb_drop_n)
        rnn1_n=BatchNormalization()(rnn1)
        rnn2=LSTM(1500,return_sequences=True)(rnn1_n)
        rnn2_n=BatchNormalization()(rnn2)
        rnn3=LSTM(1500,return_sequences=True)(rnn2_n)
        rnn3_n=BatchNormalization()(rnn3)
        proj_weights=TimeDistributed(Dense(1500))(Add()([rnn1_n,rnn2_n,rnn3_n]))
        proj_weights_gelu=Lambda(lambda x: tf.multiply(x,tf.nn.sigmoid(tf.scalar_mul(1.702,x))))(proj_weights)
        dec=Dense(vocab_size,activation="softmax",name="decision")
        dec_td=TimeDistributed(dec)(proj_weights_gelu)
        mod=tf.keras.Model(inputs=[inp],outputs=[dec_td])

        return mod



class MyModel(tf.keras.Model):

    def save(self,name,*args,**kwargs):
        #1) Save own weights
        self.save_weights(name+".h5")
        #2) Save the optimizer
        with open(name+"-optimizer.json","wt") as f:
            print(json.dumps(tf.keras.optimizers.serialize(self.optimizer)),file=f)
        with open(name+"-optimizer-weights.pickle","wb") as f:
            pickle.dump(self.optimizer.get_weights(),f)

        
class LSTMModel_5(MyModel):

    def load_weights_and_opt(self,name,*args,**kwargs):
        opt_dict=json.load(open(name+"-optimizer.json"))
        print("Optimizer config:",opt_dict,file=sys.stderr)
        optimizer=tf.keras.optimizers.deserialize(opt_dict)
        self.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",sample_weight_mode="temporal")
        self.train_on_batch(tf.convert_to_tensor([[0,0],[0,0]],dtype=tf.int64),tf.convert_to_tensor([[0,0],[0,0]],dtype=tf.int64))

        self.load_weights(name+".h5",by_name=True)
        self.optimizer.set_weights(pickle.load(open(name+"-optimizer-weights.pickle","rb")))
        print("Weights loaded.",file=sys.stderr)

    
    def __init__(self,vocab_size,training=False):
        super(LSTMModel_5, self).__init__(name='lstm-mod-5')
        #input
        self.vocab_size=vocab_size

        self.emb_layer=Embedding(input_dim=vocab_size,output_dim=1500,embeddings_initializer=tf.keras.initializers.Constant(0.01),name="input_emb")
        self.spatial_do=SpatialDropout1D(0.7,name="spatial_do_1")
        
        self.batch_norm_postemb=BatchNormalization(name="batch_norm_postemb")
        self.batch_norm_postrnn=BatchNormalization(name="batch_norm_postrnn")


        if training:
            stateful=False
        else:
            stateful=True #when we test this, we want to run a step at a time
            
        self.rnn1=LSTM(1500,return_sequences=True,stateful=stateful,name="lstm_1")
        self.rnn2=LSTM(1500,return_sequences=True,stateful=stateful,name="lstm_2")
        self.rnn3=LSTM(1500,return_sequences=True,stateful=stateful,name="lstm_3")
        self.rnn4=LSTM(1500,return_sequences=True,stateful=stateful,name="lstm_4")
        self.rnn5=LSTM(1500,return_sequences=True,stateful=stateful,name="lstm_5")

        self.proj_weights=Dense(1500,name="proj_weights_1")
        
        self.smax_dense=Dense(vocab_size,activation="softmax",name="decision")


    def call(self,inputs,training=False):
        print("TRAINING GIVEN AS",training)
        training=True
        print("CALL training=",training)
        emb=self.emb_layer(inputs)
        if training:
            emb=self.spatial_do(emb)
        emb_norm=self.batch_norm_postemb(emb)

        rnn1=self.rnn1(emb_norm)
        rnn1_bypass=Add()([emb_norm,rnn1])
        
        rnn2=self.rnn2(rnn1_bypass)
        rnn2_bypass=Add()([rnn1_bypass,rnn2])
        
        rnn3=self.rnn3(rnn2_bypass)
        rnn3_bypass=Add()([rnn2_bypass,rnn3])
        
        rnn4=self.rnn4(rnn3_bypass)
        rnn4_bypass=Add()([rnn3_bypass,rnn4])

        rnn5=self.rnn5(rnn4_bypass)
        rnn5_bypass=Add()([rnn4_bypass,rnn5])
        
        rnn5_norm=self.batch_norm_postrnn(rnn5_bypass)
        
        proj=TimeDistributed(self.proj_weights)(rnn5_norm)
        proj=Lambda(lambda x: tf.multiply(x,tf.nn.sigmoid(tf.scalar_mul(1.702,x))))(proj)
        smax=self.smax_dense(proj)
        return smax
    
        
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.vocab_size) #ummm, I don't quite know what I am doing here :D
        return tf.TensorShape(shape)
