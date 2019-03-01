import sys
import gzip

ID,FORM,LEMMA,UPOS,XPOS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)

def yield_docs_from_conllus(fnames):
    for fname in fnames:
        if fname.endswith(".gz"):
            f=gzip.open(fname,"rt",encoding="utf-8")
        else:
            f=open(fname,encoding="utf-8")
        yield from yield_docs_from_conllu(f)
        f.close()

def yield_docs_from_conllu(f):
    """Yields lists of sentences, each sentence being the usual list of lines which are lists of columns"""
    current_document=[[]]
    for line in f:
        line=line.strip()
        if not line and current_document[-1]:
            current_document.append([])
        elif line.startswith("###C:<doc"):
            current_document=[sent for sent in current_document if sent]
            if current_document:
                yield current_document
            current_document=[[]]
        elif line.startswith("#"):
            continue
        else:
            current_document[-1].append(line.split("\t"))
    else:
        current_document=[sent for sent in current_document if sent]
        if current_document:
            yield current_document
        f.close()

def batch_sentences(docs,max_batch):
    """Merges sentences as they come from docs, max_batch at a time."""
    for doc in docs:
        current_batch=[]
        for sent in doc:
            current_batch.append(sent)
            if len(current_batch)>=max_batch:
                yield current_batch
                current_batch=[]
        else:
            if current_batch:
                yield current_batch
                current_batch=[]

def flatten_sentences(batches):
    """Flattens a list of sentences into a single list of wordforms"""
    for batch in batches:
        words=list(line[FORM] for sent in batch for line in sent)
        yield words


def word_lists_from_conllus(fnames,batch_size=5):
    docs=yield_docs_from_conllus(fnames)
    batches=batch_sentences(docs,batch_size)
    word_lists=flatten_sentences(batches)
    return word_lists
    

if __name__=="__main__":
    for wl in word_lists_from_conllus(sys.argv[1:],batch_size=5):
        print(" ".join(wl))
        

