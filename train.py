import tensorflow as tf
import data
import model
import glob
import argparse
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
import os
import sys

config = tf.ConfigProto()#(log_device_placement=True)
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


class BSaver(Callback):

    def __init__(self, N, name):
        self.N = N
        self.batch = 1
        self.name=name


    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            self.model.save(self.name)
            print("\n\nSaving, having seen",self.batch,"batches\n",file=sys.stderr,flush=True)
        self.batch += 1

def get_vocab_size(vocab_file_name):
    contents=open(vocab_file_name,encoding="utf-8").read().strip()
    return len(contents.split("\n"))
            
        
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="LM train")
    parser.add_argument("--data",nargs="+",help="Data files")
    parser.add_argument("--checkpoint-dir",default=None,help="Checkpoint dir, saved every 10min")
    parser.add_argument("--lr",type=float,default=0.001,help="learning rate")
    parser.add_argument("--model-class",help="Model class from model.py")
    args=parser.parse_args()
    os.makedirs(args.checkpoint_dir,exist_ok=True)
    datafiles=args.data

    
    
    train_dataset=data.dataset_from_conllu(datafiles[:-1],"vocab.txt").prefetch(20).repeat()
    train_it=train_dataset.make_initializable_iterator() #this is needed because of the table lookup op

    dev_dataset=data.dataset_from_conllu(datafiles[-1:],"vocab.txt").shuffle(1000).take(100).repeat()
    dev_it=dev_dataset.make_initializable_iterator() #this is needed because of the table lookup op

    train_init_op=train_it.initializer
    dev_init_op=dev_it.initializer

    model_class=getattr(model,args.model_class)
    m=model_class(vocab_size=get_vocab_size("vocab.txt"))
    keras_m=m.model
    opt=tf.keras.optimizers.Adam(lr=args.lr,beta_2=0.99,amsgrad=True)
    #opt=tf.keras.optimizers.Adagrad(lr=args.lr)
    keras_m.compile(loss="sparse_categorical_crossentropy",optimizer=opt,sample_weight_mode="temporal")
    keras_m.summary()

    saver=ModelCheckpoint(os.path.join(args.checkpoint_dir,"best.rnnlm"),save_best_only=True,verbose=1)
    saver_all=ModelCheckpoint(os.path.join(args.checkpoint_dir,"epochlast.rnnlm"),verbose=1)
    #saver_security=BSaver(10,os.path.join(args.checkpoint_dir,"last.rnnlm"))

    
    with tf.keras.backend.get_session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(train_init_op)
        sess.run(dev_init_op)
        keras_m.fit(train_it,validation_data=dev_it,steps_per_epoch=10000,epochs=20000,validation_steps=100,callbacks=[saver,saver_all])
        # while True:
        #     try:
        #         elem=sess.run(next_elem)
        #     except tf.errors.OutOfRangeError:
        #         break
        #     print(elem[0].shape,elem[1].shape)
        

            
