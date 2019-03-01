import gzip
import itertools
import collections
import tensorflow as tf
import sys
import numpy as np
import random
import multiprocessing as mp
import sentencepiece as sp
import conllu_utils

class DataPipeline:

    def __init__(self,batch_size):
        self.batch_size=batch_size

    def yield_wordlists_from_conllu(self, conllu_names):
        random.shuffle(conllu_names)
        for counter,wordlist in enumerate(conllu_utils.word_lists_from_conllus(conllu_names,self.batch_size)):
            yield (wordlist,) #...we need to yield tuples, even if they only had one element, they need to be explicitly there for the dataset interface to work
            if (counter+1) % 10000==0:
                print("\n\nSeen",counter*self.batch_size,"sentences\n",file=sys.stderr,flush=True)

    def get_vocab_size(self, vocab_file_name):
        with open(vocab_file_name, "rt", encoding="utf-8") as f:
            contents=f.read().strip()
        return len(contents.split("\n"))


    def dataset_from_conllu_names(self, conllu_names):
        sentences=self.indexed_dataset(conllu_names).prefetch(20)
        
        sentences=sentences.filter(lambda seq: tf.shape(seq)[0]<=150)
        sentences=sentences.filter(lambda seq: tf.shape(seq)[0]>=5)
        sentences_xy=sentences.map(lambda seq: (seq[:-1],seq[1:]))
        
        elem_len_func=lambda x,y: tf.shape(x)[0] #length of sequence is its first shape element
        bucket_boundaries=np.arange(5,155,20) #5,10,15,... generate one-extra so we can get bucket_batch_sizes easily
        bucketeer=tf.data.experimental.bucket_by_sequence_length(element_length_func=elem_len_func,
                                                                 bucket_boundaries=bucket_boundaries[:-1], #one less, since bucket_batch_sizes must be one longer than the boundaries
                                                                 bucket_batch_sizes=[4000//i for i in bucket_boundaries])
        sentences_bucketed=sentences_xy.apply(bucketeer)
        sentences_bucketed=sentences_bucketed.map(lambda x,y: (x,tf.expand_dims(y,-1)))
        return sentences_bucketed

class TokenDataPipeline(DataPipeline):

    def __init__(self, vocab_file_name, batch_size=5):
        super(TokenDataPipeline,self).init(batch_size)
        self.vocab_name=vocab_file_name
        self.vocab_size=self.get_vocab_size(vocab_file_name)

    def indexed_dataset(self,conllu_names):
        #TODO: switch to tf.contrib.lookup.index_table_from_file ?
        vocab_table_init=tf.contrib.lookup.TextFileInitializer(self.vocab_name,tf.string,0,tf.int64,1,delimiter="\t") 
        vocab_lookup=tf.contrib.lookup.HashTable(vocab_table_init,1,"vocab_lookup","vocab_lookup_op") #1 is the index of <UNK> in the vocabulary. 0 is pad and 2 is <EOS>
        
        shape=tf.TensorShape([None]) #sentence generator produces sequences of words of arbitrary length, so this is their shape
        sentences=tf.data.Dataset.from_generator(lambda: self.yield_wordlists_from_conllu(conllu_names,batch_size),output_types=(tf.string,),output_shapes=(shape,))
        sentences_num=sentences.map(vocab_lookup.lookup)
        return sentences_num
        
class SubwordDataPipeline(DataPipeline):

    def __init__(self, subword_model_name, batch_size=5):
        super(SubwordDataPipeline,self).__init__(batch_size)
        self.subword_model=sp.SentencePieceProcessor()
        self.subword_model.Load("{name}.model".format(name=subword_model_name))
        # if add_markers:
        #     self.subword_model.SetEncodeExtraOptions(extra_option="bos:eos")
        self.vocab_size=self.get_vocab_size(subword_model_name+".vocab")

    def indexed_dataset(self,conllu_names):
        shape=tf.TensorShape([None]) #sentence generator produces sequences of words of arbitrary length, so this is their shape
        word_lists=self.yield_wordlists_from_conllu(conllu_names)
        sentences_num=tf.data.Dataset.from_generator(lambda: self.index(word_lists), output_types=(tf.int32,), output_shapes=(shape,))
        return sentences_num

    def index(self,wordlists):
        for wordlist in wordlists:
            ids=self.subword_model.EncodeAsIds(" ".join(wordlist[0]))
            yield (ids,)


if __name__=="__main__":
    sentences=yield_sents_from_conllu("/dev/stdin",add_markers=False)
    sentences=(s[0] for s in sentences) #sentences returns 1-tuple so this needs to be compensated for
    words=itertools.chain.from_iterable(sentences)
    words_10M=itertools.islice(words,10000000) #make dictionary based on the first 10M words
    c=collections.Counter(words_10M)
    print("<PAD>",0,sep="\t")
    print("<UNK>",1,sep="\t")
    print("<BOS>",2,sep="\t")
    print("<EOS>",3,sep="\t")
    index_offset=4 #this compensates for these three special symbols
    for idx,(word,count) in enumerate(c.most_common(100000)): #take 100K words, drop rest
        print(word,idx+index_offset,sep="\t")
    
    
        
    
