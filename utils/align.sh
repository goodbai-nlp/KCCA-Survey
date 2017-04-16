#!/bin/sh
export Frdict="/home/xfbai/corpus/Bilingual_dict/parallel.fwdxbwd-dict.fr-en"
export Zhdict="/home/xfbai/corpus/Bilingual_dict/google_dict.zh-en"
export Frvec="/home/xfbai/tmpvec/new_embedding_size200.fr"
export Zhvec="/home/xfbai/tmpvec/new_embedding_size200.zh"
export Envec="/home/xfbai/tmpvec/new_embedding_size200.en"
export FrWord="/home/xfbai/tmpvec/fr_wordCount.txt"
export ZhWord="/home/xfbai/tmpvec/zh_wordCount.txt"
export EnWord="/home/xfbai/tmpvec/en_wordCount.txt"
set -e
echo "Aligning vectors..."
python alignVectors.py -w1 $Zhvec -w2 $Envec -w3 $ZhWord -w4 $EnWord -a $Zhdict -o Out
