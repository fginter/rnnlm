import sentencepiece as sp
import argparse
import sys
import os
import conllu_utils

"""
Run: zcat data.conllu.gz | python subword_trainer.py --model_name subwordvocab
"""

ID,FORM,LEMMA,FEAT,UPOS,XPOS,HEAD,DEPREL,DEPS,MISC=range(10)

def train(args):

    with open("text.txt.tmp", "wt", encoding="utf-8") as f:
        for i,sent in enumerate(conllu_utils.word_lists_from_conllus(["/dev/stdin"])):
            print(" ".join(sent), file=f)
            if i>args.max_sents:
                break

    sp.SentencePieceTrainer.train('--model_prefix={name} --input=text.txt.tmp --vocab_size={size} --model_type=bpe --pad_id=0 --bos_id=2 --eos_id=3 --unk_id=1'.format(name=args.model_name, size=args.vocab_size))

    os.remove("text.txt.tmp") # this ugly hack should be removed...

def test(args):

    model=sp.SentencePieceProcessor()
    model.Load("{name}.model".format(name=args.model_name))
    #model.SetEncodeExtraOptions(extra_option="bos:eos")

    for i,sent in enumerate(conllu_utils.word_lists_from_conllus(["/dev/stdin"])):
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
    parser.add_argument("--segment", default=False, action="store_true", help="Test segmentation on the given data")
    args=parser.parse_args()

    if args.segment:
        test(args)
        sys.exit()

    train(args)
