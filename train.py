import tensorflow as tf
import data
import model
import glob
import argparse
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
import os
import sys
import datetime



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
            
        
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="LM train")
    parser.add_argument("--data",nargs="+",help="Data files")
    parser.add_argument("--checkpoint-dir",default=None,help="Checkpoint dir, saved every 10min")
    parser.add_argument("--restart",default=None,help="Restart from a given checkpoint file")
    parser.add_argument("--lr",type=float,default=0.001,help="learning rate")
    parser.add_argument("--steps-per-epoch",type=int,default=5000,help="Batches in one epoch. Default: %(default)d")
    parser.add_argument("--epochs",type=int,default=10000,help="Epochs. Default: %(default)d")
    parser.add_argument("--validation-steps",type=int,default=20,help="Validation batches. Default: %(default)d")
    parser.add_argument("--model-class",help="Model class from model.py")
    parser.add_argument("--items-per-batch",type=int,default=16000,help="items (words, s-pieces) in one batch default: %(default)d")
    parser.add_argument("--pipeline", type=str, required=True,choices=["subword", "token"], help="Data pipeline type (subword or token)")
    parser.add_argument("--vocab", type=str, required=True, help="Vocab file name (for token data pipeline) or subword model name (for subword data pipeline, subword model name must be without .model or .vocab extension)")
    args=parser.parse_args()
    os.makedirs(args.checkpoint_dir,exist_ok=True)
    datafiles=args.data

    print("Data Pipeline:", args.pipeline, file=sys.stderr)
    if args.pipeline=="subword":
        data_pipeline=data.SubwordDataPipeline(subword_model_name=args.vocab)
    else:
        data_pipeline=data.TokenDataPipeline(vocab=args.vocab)
    
    train_dataset=data_pipeline.dataset_from_conllu_names(datafiles[:-1], items_per_batch=args.items_per_batch).prefetch(20).repeat()
    train_it=train_dataset.make_initializable_iterator() #this is needed because of the table lookup opn

    dev_dataset=data_pipeline.dataset_from_conllu_names(datafiles[-1:], items_per_batch=args.items_per_batch).shuffle(1000).take(100).repeat()
    dev_it=dev_dataset.make_initializable_iterator() #this is needed because of the table lookup op

    train_init_op=train_it.initializer
    dev_init_op=dev_it.initializer


    run_timestamp=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_class=getattr(model,args.model_class)
    if args.restart:
        print("Restarting from",args.restart,file=sys.stderr,flush=True)
        keras_m=model_class(vocab_size=data_pipeline.vocab_size,training=True)
        keras_m.load_weights_and_opt(args.restart,by_name=True)
        
    else:
        keras_m=model_class(vocab_size=data_pipeline.vocab_size,training=True)
        opt=tf.keras.optimizers.Adam(lr=args.lr,beta_2=0.99,amsgrad=True)
        #opt=tf.keras.optimizers.Adagrad(lr=args.lr)
        keras_m.compile(loss="sparse_categorical_crossentropy",optimizer=opt,sample_weight_mode="temporal")
        #keras_m.save(args.checkpoint_dir+"/initial.h5")
        #print(tf.keras.optimizers.serialize(keras_m.optimizer))
        #keras_m.summary()

    saver=ModelCheckpoint(os.path.join(args.checkpoint_dir,"best.rnnlm"),save_best_only=True,verbose=1)
    epoch_filename="epoch.{tstamp}.{{epoch:05d}}.last.rnnlm".format(tstamp=run_timestamp)
    print("Will save epoch models named as",epoch_filename,file=sys.stderr,flush=True)
    saver_all=ModelCheckpoint(os.path.join(args.checkpoint_dir,epoch_filename),verbose=1)
    #saver_security=BSaver(10,os.path.join(args.checkpoint_dir,"last.rnnlm"))

    
    with tf.keras.backend.get_session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(train_init_op)
        sess.run(dev_init_op)
        print("START FIT")
        keras_m.fit(train_it,validation_data=dev_it,steps_per_epoch=args.steps_per_epoch,epochs=args.epochs,validation_steps=args.validation_steps,callbacks=[saver,saver_all])
        print("DONE FIT")
        # while True:
        #     try:
        #         elem=sess.run(next_elem)
        #     except tf.errors.OutOfRangeError:
        #         break
        #     print(elem[0].shape,elem[1].shape)
        

            
