#!/bin/sh
export Frdict="/home/xfbai/corpus/Bilingual_dict/parallel.fwdxbwd-dict.fr-en"
export Foreignvec="/home/xfbai/tmpvec/new_embedding_size40.fr"
export Envec="/home/xfbai/tmpvec/new_embedding_size40.en"
export ForeignWord="/home/xfbai/tmpvec/fr_wordCount.txt"
export EnWord="/home/xfbai/tmpvec/en_wordCount.txt"
set -e
echo "Aligning vectors..."
python alignVectors.py -w1 $Foreignvec -w2 $Envec -w3 $ForeignWord -w4 $EnWord -a $Frdict -o Out
