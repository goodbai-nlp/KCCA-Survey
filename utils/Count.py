#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: Count.py.py
@time: 17-3-31 下午3:18
"""
#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: Count.py.py
@time: 17-3-31 下午3:18
"""
from collections import Counter
import time
import argparse
import sys
reload(sys)
sys.setdefaultencoding('utf8')
def Count(originFile,newFile):
    '''
    统计语料中每个词的词频
    :return: 词频字典 
    '''
    wordDict = Counter()
    f1 = open(originFile,'rb')
    f2 = open(newFile,'wb')
    if f1:
        for line in f1:
            sentence = line.strip().split(' ')
	    if len(sentence):
                wordDict.update(sentence)
    else:
        print "failed to open file!"
    mydict=[(key,value) for key,value in wordDict.items() if value >5]
    print "Number of counts >5 words:",len(mydict)
    if f2:
        for key,value in wordDict.items():
            line = "%s:%s\n" % (key, str(value))
            f2.write(line.encode('utf-8'))
    else:
        print "failed to open output file"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w1", "--originFile", type=str, help="Origin Vector file")
    parser.add_argument("-o", "--outputFile", type=str, help="Output word count file")
    args = parser.parse_args()
    print "\nstart generate word count file.....\n"
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    Count(args.originFile,args.outputFile)
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
