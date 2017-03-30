export output_embeddings="new_location" #"/usr1/home/wammar/cca-embeddings/all_languages.cca.window_5+iter_10+size_40+threads_16"
export temp="/home/xfbai/tmpvec/"
export utils="/home/xfbai/utils/"
export word2vec="/home/xfbai/tools/word2vec"

# create temp dir
mkdir $temp

# remove old embeddings if any
rm $output_embeddings
printf '\n\b\b\b\b\b\b\b\b%s\n' `date +%T`
# process en
export corpus_en="/home/xfbai/corpus/monolingual/mono.tok.lc.en"
$word2vec/word2vec -train $corpus_en -window 5 -iter 10 -size 40 -threads 16 -output $temp/embeddings.en

# process fr
export corpus_fr="/home/xfbai/corpus/monolingual/mono.tok.lc.fr"
$word2vec/word2vec -train $corpus_fr -window 5 -iter 10 -size 40 -threads 16 -output $temp/embeddings.fr
export corpus_fr="/home/xfbai/corpus/monolingual/mono.tok.lc.de"
$word2vec/word2vec -train $corpus_fr -window 5 -iter 10 -size 40 -threads 16 -output $temp/embeddings.de
printf '\n\b\b\b\b\b\b\b\b%s\n' `date +%T`
