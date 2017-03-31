import sys
import argparse
import numpy
import gzip
import math
OutputDir = "/home/xfbai/mywork/git/KCCA-Experiment/data"
''' 提取字典里对应的词的双语的词向量，并正则化，输出为两个文件'''
'''用法： python alignVectors.py -w1 ForiegnVecFile.txt -w2 EnVecFile.txt -a WordAlignFile -o OutFileFirstName'''
def read_word_vectors(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')

  for lineNum, line in enumerate(fileObject):
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)        
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
            
  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

def save_orig_subset_and_aligned(outFileName, lang2WordVectors, lang1AlignedVectors):
  outFile = open(OutputDir+outFileName+'_en_new_aligned.txt','w') #英语
  for word in lang1AlignedVectors:
    outFile.write(word+' '+' '.join([str(val) for val in lang2WordVectors[word]])+'\n')
  outFile.close()
  
  outFile = open(OutputDir+outFileName+'_foreign_new_aligned.txt','w') #法语
  for word in lang1AlignedVectors:
    outFile.write(word+' '+' '.join([str(val) for val in lang1AlignedVectors[word]])+'\n')
  outFile.close()
  
def get_aligned_vectors(wordAlignFile, lang1WordVectors, lang2WordVectors):
  alignedVectors = {}
  lenLang1Vector = len(lang1WordVectors[lang1WordVectors.keys()[0]])
  for line in open(wordAlignFile, 'r'):
    lang1Word, lang2Word = line.strip().split(" ||| ")
    if lang2Word not in lang2WordVectors: continue
    if lang1Word not in lang1WordVectors: continue
    alignedVectors[lang2Word] = numpy.zeros(lenLang1Vector, dtype=float)
    alignedVectors[lang2Word] += lang1WordVectors[lang1Word]

  sys.stderr.write("No. of aligned vectors found: "+str(len(alignedVectors))+'\n')      
  return alignedVectors


if __name__=='__main__':
    
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--wordaligncountfile", type=str, help="Word alignment count file")
  parser.add_argument("-w1", "--wordproj1", type=str, help="Word proj of lang1")
  parser.add_argument("-w2", "--wordproj2", type=str, help="Word proj of lang2")
  parser.add_argument("-o", "--outputfile", type=str, help="Output file for storing aligned vectors")
    
  args = parser.parse_args()
  lang1WordVectors = read_word_vectors(args.wordproj1)
  lang2WordVectors = read_word_vectors(args.wordproj2)
    
  lang1AlignedVectors = get_aligned_vectors(args.wordaligncountfile, lang1WordVectors, lang2WordVectors)
  save_orig_subset_and_aligned(args.outputfile, lang2WordVectors, lang1AlignedVectors)