import sentencepiece as sp
import argparse
import sys
import os

"""
Run: zcat data.conllu.gz | python subword_trainer.py --model_name subwordvocab
"""

ID,FORM,LEMMA,FEAT,UPOS,XPOS,HEAD,DEPREL,DEPS,MISC=range(10)

def yield_sents_from_conllu(f):
    current_sent=[]
    for line in f:
        line=line.strip()
        if not line and current_sent:
            words=list(cols[FORM].lower() for cols in current_sent)
            yield words
            current_sent=[]
        elif line.startswith("#"):
            continue
        else:
            current_sent.append(line.split("\t"))
    else:
        f.close()

def train(args):

    with open("text.txt.tmp", "wt", encoding="utf-8") as f:
        for i,sent in enumerate(yield_sents_from_conllu(sys.stdin)):
            print(" ".join(sent), file=f)
            if i>args.max_sents:
                break

    sp.SentencePieceTrainer.train('--model_prefix={name} --input=text.txt.tmp --vocab_size={size} --model_type=bpe --pad_id=0 --bos_id=2 --eos_id=3 --unk_id=1'.format(name=args.model_name, size=args.vocab_size))

    os.remove("text.txt.tmp") # this ugly hack should be removed...

def test(args):

    model=sp.SentencePieceProcessor()
    model.Load("{name}.model".format(name=args.model_name))
    model.SetEncodeExtraOptions(extra_option="bos:eos")

    for i,sent in enumerate(yield_sents_from_conllu(["/dev/stdin"],add_markers=False)):
            print("Pieces:"," ".join(model.EncodeAsPieces(" ".join(sent[0]))))
            print("Ids:"," ".join(str(ids) for ids in model.EncodeAsIds(" ".join(sent[0]))))
            print()
            if i>10:
                break

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Subword vocab trainer")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--vocab_size", type=int, default=20000, help="Vocab size")
    parser.add_argument("--max_sents", type=int, default=1000000, help="How many sentences to use for training")
    parser.add_argument("--segment", default=False, action="store_true", help="Test segmentation for given the data")
    args=parser.parse_args()

    if args.segment:
        test(args)
        sys.exit()

    train(args)
