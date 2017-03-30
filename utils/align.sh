#!/bin/sh
export Frdict = "/home/xfbai/corpus/Bilingual_dict/parallel.fwdxbwd-dict.fr-en"
export Foreignvec = "/home/xfbai/tmpvec/embeddings.fr"
export Envec = "/home/xfbai/tmpvec/embeddings.en"
set -e
echo "Aligning vectors..."
python alignVectors.py -w1 $Foreignvec -w2 $Envec -a $Frdict -o Out
