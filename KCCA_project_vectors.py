#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: KCCA_project_vectors.py
@time: 17-3-29 下午10:32
"""
import numpy as np
import time
import commands
from sklearn import preprocessing
from myKcca import KCCA

datadir = "/home/xfbai/mywork/git/KCCA-Experiment/data/"
OutputDir="/home/xfbai/mywork/git/KCCA-Experiment/Output/"
#origForeignVecFile = "/home/xfbai/tmpvec/new_embedding_size200.fr"
#origForeignVecFile = "/home/xfbai/tmpvec/new_embedding_size200.zh"
#origForeignVecFile = "/home/xfbai/tmpvec/new_embedding_size200.de"
#origForeignVecFile = "/home/xfbai/tmpvec/new_embedding_size200.fi"
#origForeignVecFile = "/home/xfbai/tmpvec/new_embedding_size200.hu"
origForeignVecFile = "/home/xfbai/tmpvec/new_embedding_size200.cs"
origEnVecFile = "/home/xfbai/tmpvec/new_embedding_size200.en"
subsetEnVecFile = datadir+"Out_en_new_aligned.txt"
subsetForeignVecFile = datadir+"Out_foreign_new_aligned.txt"
outputEnFile = OutputDir+"KCCA_en_out.txt"
outputForeignFile = OutputDir+"KCCA_foreign_out.txt"
paramFile = OutputDir+"param.txt"
def project_vectors(origForeignVecFile,origEnVecFile,subsetEnVecFile,subsetForeignVecFile,outputEnFile,outputForeignFile,NUMCC=20):
    '''
    将词典的向量输入到KCCA中，生成投影向量，再生成双语向量
    :param origForeignVecFile: 外语向量矩阵
    :param origEnVecFile: 英语向量矩阵
    :param subsetEnVecFile: 词典中的英语向量矩阵
    :param subsetForeignVecFile: 词典中的外语向量矩阵
    :param outputEnFile: 重新获得的英语词向量
    :param outputForeignFile: 重新获得的外语词向量
    :param truncRatio: 模型的训练系数
    '''

    '''数据读入，处理掉开头的英文单词，只保留词向量'''
    tmp = np.loadtxt(origEnVecFile,dtype=np.str,delimiter=' ')
    origEnVecs = tmp[:,1:].astype(np.float)
    tmp2 = np.loadtxt(origForeignVecFile, dtype=np.str, delimiter=' ')
    origForeignVecs= tmp2[:, 1:].astype(np.float)
    tmp3 = np.loadtxt(subsetEnVecFile, dtype=np.str,delimiter=' ')
    subsetEnVecs = tmp3[:, 1:].astype(np.float)
    tmp4 =np.loadtxt(subsetForeignVecFile,dtype=np.str,delimiter=' ')
    subsetForeignVecs = tmp4[:,1:].astype(np.float)

    '''预处理，使每行正则化'''
    origEnVecs=preprocessing.scale(origEnVecs)
    origForeignVecs=preprocessing.scale(origForeignVecs)

    '''训练模型'''
    x1 = subsetEnVecs
    x2 = subsetForeignVecs
    resDict={}
    for i in range(3,4):
        for j in range(3,4):
            #tmpvec=[2,3,0.9,0.8]
            #for k in tmpvec[::]:
            for k in range(1,2,1):
                kcca=KCCA('rbf','rbf',regularization=10**-2,gamma1=10**-4,gamma2=10**-4,n_jobs=-1,n_components=NUMCC).fit(x1,x2)
                # print kcca.corrs
                y1, y2 = kcca.transform(origEnVecs, origForeignVecs)
                origEnVecsProjected = preprocessing.scale(y1)
                origEnVecsProjected =np.column_stack((tmp[:,:1],origEnVecsProjected.astype(np.str)))
                origForeignVecsProjected = preprocessing.scale(y2)
                origForeignVecsProjected = np.column_stack((tmp2[:, :1],origForeignVecsProjected.astype(np.str)))
                np.savetxt(outputEnFile,origEnVecsProjected,fmt="%s",delimiter=' ')
                np.savetxt(outputForeignFile,origForeignVecsProjected,fmt="%s",delimiter=' ')
                a,b  =   commands.getstatusoutput( 'python /home/xfbai/mywork/git/KCCA-Experiment/evalution/qvec/qvec_cca2.py')
                score= float(b.strip().split()[-1])
                if(score<0.3):
                    print "score = ",score,"passed"
                    continue
                #c,d = commands.getstatusoutput('bash run.sh evaluateBiDict fr /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_en_out.txt /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_foreign_out.txt')
                #c,d = commands.getstatusoutput('bash run.sh evaluateBiDict zh /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_en_out.txt /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_foreign_out.txt')
                #c,d = commands.getstatusoutput('bash run.sh evaluateBiDict de /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_en_out.txt /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_foreign_out.txt')
                #c,d = commands.getstatusoutput('bash run.sh evaluateBiDict fi /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_en_out.txt /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_foreign_out.txt')
                #c,d = commands.getstatusoutput('bash run.sh evaluateBiDict hu /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_en_out.txt /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_foreign_out.txt')
                c,d = commands.getstatusoutput('bash run.sh evaluateBiDict cs /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_en_out.txt /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_foreign_out.txt')
                print b
                print d
                ff = open(paramFile,'a')
                ttmpstr = str((10**-4,10**-4,10**-2))+' '+b+d+"\n"
                ff.write(ttmpstr)
                ff.close()
    print "Work Finished"


if __name__ == "__main__":
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print "training model..."
    project_vectors(origForeignVecFile, origEnVecFile, subsetEnVecFile, subsetForeignVecFile,outputEnFile,outputForeignFile,100)
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
