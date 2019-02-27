import tensorflow as tf
import data
import model
import glob
import argparse
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
import os
import sys

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="LM train")
    parser.add_argument("--checkpoint",default=None,help="Checkpoint dir, saved every 10min")
    args=parser.parse_args()

    m=tf.keras.models.load_model(args.checkpoint)
    m.summary()
    
