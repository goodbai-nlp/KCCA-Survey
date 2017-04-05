export output_embeddings="new_location" #"/usr1/home/wammar/cca-embeddings/all_languages.cca.window_5+iter_10+size_40+threads_16"
export temp="/home/xfbai/tmpvec/"
export utils="/home/xfbai/mywork/git/KCCA-Experiment/utils"
export word2vec="/home/xfbai/tools/word2vec"

# create temp dir
mkdir $temp

# remove old embeddings if any
rm $output_embeddings
printf '\n\b\b\b\b\b\b\b\b%s\n' `date +%T`
# process en
export corpus_en="/home/xfbai/corpus/monolingual/mono.tok.lc.en"
$word2vec/word2vec -train $corpus_en -window 5 -iter 10 -size 40 -threads 16 -output $temp/embeddings_size40.en
python $utils/Count.py -w1 $corpus_en -o $temp/en_wordCount.txt
python $utils/Predeal.py -w1 $temp/embedding_size40.en -w2 $temp/en_wordCount.txt -o new_embedding_size40.en
# process fr
export corpus_fr="/home/xfbai/corpus/monolingual/mono.tok.lc.fr"
$word2vec/word2vec -train $corpus_fr -window 5 -iter 10 -size 40 -threads 16 -output $temp/embeddings_size40.fr
python $utils/Count.py -w1 $corpus_fr -o $temp/fr_wordCount.txt
python $utils/Predeal.py -w1 $temp/embedding_size40.fr -w2 $temp/fr_wordCount.txt -o new_em
bedding_size40.fr
# process de
export corpus_de="/home/xfbai/corpus/monolingual/mono.tok.lc.de"
# $word2vec/word2vec -train $corpus_de -window 5 -iter 10 -size 40 -threads 16 -output $temp/embeddings_size40.de
python $utils/Count.py -w1 $corpus_de -o $temp/de_wordCount.txt
python $utils/Predeal.py -w1 $temp/embedding_size40.de -w2 $temp/de_wordCount.txt -o new_em
bedding_size40.de
printf '\n\b\b\b\b\b\b\b\b%s\n' `date +%T`
