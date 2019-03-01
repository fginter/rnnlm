import gzip
import itertools
import collections
import tensorflow as tf
import sys
import numpy as np
import random
import sentencepiece as sp

ID,FORM,LEMMA,FEAT,UPOS,XPOS,HEAD,DEPREL,DEPS,MISC=range(10)

def yield_sentcolumns_from_conllu(conllu_name):
    """conllu into list of columns, one per word"""
    if conllu_name.endswith(".gz"):
        f=gzip.open(conllu_name,"rt",encoding="utf-8")
    else:
        f=open(conllu_name,encoding="utf-8")
    current_sent=[]
    for line in f:
        line=line.strip()
        if not line and current_sent:
            yield current_sent
            current_sent=[]
        elif line.startswith("#"):
            continue
        else:
            current_sent.append(line.split("\t"))
    else:
        f.close()
        

class TokenDataPipeline(object):

    def __init__(self, vocab=None, add_markers=True):
        self.vocab_name=vocab
        self.add_markers=add_markers
        self.vocab_size=self.get_vocab_size(vocab)


    def get_vocab_size(self, vocab_file_name):
        with open(vocab_file_name, "rt", encoding="utf-8") as f:
            contents=f.read().strip()
        return len(contents.split("\n"))

    def yield_sents_from_conllu(self, conllu_names):
        random.shuffle(conllu_names)
        for conllu_name in conllu_names:
            for counter,sent in enumerate(yield_sentcolumns_from_conllu(conllu_name)):
                words=list(cols[FORM].lower() for cols in sent) #todo this lowercase is stupid
                if self.add_markers:
                    words=["<BOS>"]+words+["<EOS>"]
                yield (words,) #...we need to yield tuples, even if they only had one element, they need to be explicitly there for the dataset interface to work
                if (counter+1) % 10000==0:
                    print("\n\nSeen",counter,"sentences\n",file=sys.stderr,flush=True)


    def dataset_from_conllu(self, conllu_name):
        #TODO: switch to tf.contrib.lookup.index_table_from_file ?
        vocab_table_init=tf.contrib.lookup.TextFileInitializer(self.vocab_name,tf.string,0,tf.int64,1,delimiter="\t") 
        vocab_lookup=tf.contrib.lookup.HashTable(vocab_table_init,1,"vocab_lookup","vocab_lookup_op") #1 is the index of <UNK> in the vocabulary. 0 is pad and 2 is <EOS>

        shape=tf.TensorShape([None]) #sentence generator produces sequences of words of arbitrary length, so this is their shape
        sentences=tf.data.Dataset.from_generator(lambda: self.yield_sents_from_conllu(conllu_name),output_types=(tf.string,),output_shapes=(shape,))
        sentences=sentences.filter(lambda seq: tf.shape(seq)[0]<=35)
        sentences=sentences.filter(lambda seq: tf.shape(seq)[0]>=5)
        sentences_num=sentences.map(vocab_lookup.lookup)
        sentences_xy=sentences_num.map(lambda seq: (seq[:-1],seq[1:]))
        
        elem_len_func=lambda x,y: tf.shape(x)[0] #length of sequence is its first shape element
        bucket_boundaries=np.arange(5,35,5) #5,10,15,... generate one-extra so we can get bucket_batch_sizes easily
        bucketeer=tf.data.experimental.bucket_by_sequence_length(element_length_func=elem_len_func,
                                                                 bucket_boundaries=bucket_boundaries[:-1], #one less, since bucket_batch_sizes must be one longer than the boundaries
                                                                 bucket_batch_sizes=[2000//i for i in bucket_boundaries])
        sentences_bucketed=sentences_xy.apply(bucketeer)
        sentences_bucketed=sentences_bucketed.map(lambda x,y: (x,tf.expand_dims(y,-1)))
        return sentences_bucketed



class SubwordDataPipeline(object):

    def __init__(self, subword_model=None, add_markers=True):

        self.subword_model=None
        self.vocab_size=None
        if subword_model:
            self.subword_model=sp.SentencePieceProcessor()
            self.subword_model.Load("{name}.model".format(name=subword_model))
            if add_markers:
                self.subword_model.SetEncodeExtraOptions(extra_option="bos:eos")
            self.vocab_size=self.get_vocab_size(subword_model)
        else:
            raise(NotImplementedError)


    def get_vocab_size(self, subword_model):
        with open("{name}.vocab".format(name=subword_model), "rt", encoding="utf-8") as f:
            contents=f.read().strip()
        return len(contents.split("\n"))

    def yield_sents_from_conllu(self, conllu_names):
        random.shuffle(conllu_names)

        for conllu_name in conllu_names:
            for counter,sent in enumerate(yield_sentcolumns_from_conllu(conllu_name)):
                words=" ".join([cols[FORM].lower() for cols in sent]) #todo this lowercase is stupid
                ids=self.subword_model.EncodeAsIds(words)
                yield (ids,) #...we need to yield tuples, even if they only had one element, they need to be explicitly there for the dataset interface to work
                if (counter+1) % 10000==0:
                    print("\n\nSeen",counter,"sentences\n",file=sys.stderr,flush=True)


    def dataset_from_conllu(self, conllu_name):

        shape=tf.TensorShape([None]) #sentence generator produces sequences of word ids of arbitrary length, so this is their shape
        sentences=tf.data.Dataset.from_generator(lambda: self.yield_sents_from_conllu(conllu_name), output_types=(tf.int32,), output_shapes=(shape,))
        sentences=sentences.filter(lambda seq: tf.shape(seq)[0]<=35)
        sentences=sentences.filter(lambda seq: tf.shape(seq)[0]>=5)

        sentences_xy=sentences.map(lambda seq: (seq[:-1],seq[1:]))
        
        elem_len_func=lambda x,y: tf.shape(x)[0] #length of sequence is its first shape element
        bucket_boundaries=np.arange(5,35,5) #5,10,15,... generate one-extra so we can get bucket_batch_sizes easily
        bucketeer=tf.data.experimental.bucket_by_sequence_length(element_length_func=elem_len_func,
                                                                 bucket_boundaries=bucket_boundaries[:-1], #one less, since bucket_batch_sizes must be one longer than the boundaries
                                                                 bucket_batch_sizes=[2000//i for i in bucket_boundaries])
        sentences_bucketed=sentences_xy.apply(bucketeer)
        sentences_bucketed=sentences_bucketed.map(lambda x,y: (x,tf.expand_dims(y,-1)))
        return sentences_bucketed

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
    
    
        
    
