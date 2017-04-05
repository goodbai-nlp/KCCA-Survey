#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: Predeal.py
@time: 17-4-5 下午3:23
"""
import argparse

wordCountDir = ""
originWordVec = ""
OutPutDir = ""

def Predeal(originFile,wordCountFile,newFile):
    wordDict = {}
    for line in open(wordCountFile):
        key,value = line.strip().split(':')
        wordDict[key]=int(value)
    file1 = open(originFile, 'rb')
    file2 = open(newFile, 'wb')
    i=0
    if (file1):
        for line in file1:
            if i!=0:
                tmp = line.strip().split(' ')
                if (tmp[0] in wordDict.keys() and wordDict[tmp[0]]>5):
                    ttmp = line.strip()+"\n"
                    file2.write(ttmp)
            i+=1
    else:
        print "Failed to open"
    file1.close()
    file2.close()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w1", "--originFile", type=str, help="Origin Vector file")
    parser.add_argument("-w2", "--wordCountFile", type=str, help="Word Count file")
    parser.add_argument("-o", "--outputfile", type=str, help="Output file for predealed vectors")
    args = parser.parse_args()
    Predeal(args.originFile,args.wordCountFile,args.outputfile)