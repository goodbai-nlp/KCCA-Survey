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
import sys
reload(sys)
sys.setdefaultencoding('utf8')
BaseDir="/home/jackbai/corpus/monolingual/"
OriginFile = ["mono.tok.lc.de","mono.tok.lc.en","mono.tok.lc.fr"]
WordCountFile = "_WordCount.txt"
def Count():
    '''
    统计语料中每个词的词频
    :return: 词频字典 
    '''
    for item in OriginFile:
        print item
        lang = item[-2:]
        wordDict = Counter()
        f1 = open(BaseDir+item,'rb')
        f2 = open(BaseDir+lang+WordCountFile,'wb')
        if f1:
            for line in f1:
                sentence = line.strip().split(' ')
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
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    Count()
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
