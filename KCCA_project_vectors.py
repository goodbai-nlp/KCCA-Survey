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
from sklearn import preprocessing
from PyKCCA import KCCA
from PyKCCA.kernels import GaussianKernel
from PyKCCA.kernels import DiagGaussianKernel
from PyKCCA.kernels import PolyKernel

datadir = "/home/xfbai/mywork/git/KCCA-Experiment/data/"
OutputDir="/home/xfbai/mywork/git/KCCA-Experiment/Output/"
origForeignVecFile = "/home/xfbai/tmpvec/embeddings.fr"
origForeignVecFileNew = "/home/xfbai/tmpvec/new_embeddings.fr"
origEnVecFile = "/home/xfbai/tmpvec/embeddings.en"
origEnVecFileNew = "/home/xfbai/tmpvec/new_embeddings.en"
subsetEnVecFile = datadir+"Out_en_new_aligned.txt"
subsetForeignVecFile = datadir+"Out_foreign_new_aligned.txt"

outputEnFile = OutputDir+"KCCA_en_out.txt"
outputForeignFile = OutputDir+"KCCA_foreign_out.txt"

def project_vectors(origForeignVecFile,origEnVecFile,subsetEnVecFile,subsetForeignVecFile,outputEnFile,outputForeignFile,NUMCC=40):
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
    tmp3 = np.loadtxt(subsetEnVecFile, dtype=np.str, delimiter=' ')
    subsetEnVecs = tmp3[:, 1:].astype(np.float)
    tmp4 = np.loadtxt(subsetForeignVecFile,dtype=np.str,delimiter=' ')
    subsetForeignVecs = tmp4[:, 1:].astype(np.float)

    '''预处理，使每行正则化'''
    origEnVecs=preprocessing.scale(origEnVecs)
    origForeignVecs=preprocessing.scale(origForeignVecs)
    subsetEnVecs = preprocessing.scale(subsetEnVecs)
    subsetForeignVecs = preprocessing.scale(subsetForeignVecs)

    '''训练CCA'''
    x1 = subsetEnVecs
    x2 = subsetForeignVecs
    kernel = GaussianKernel(sigma=1.0)
    cca = KCCA(kernel, kernel,
               regularization=1e-5,
               decomp='icd',
               method='simplified_hardoon_method',
               scaler1=lambda x: x,
               scaler2=lambda x: x,
               SSnum=NUMCC).fit(x1, x2)
    print cca.beta_
    y1, y2 = cca.transform(origEnVecs, origForeignVecs)
    origEnVecsProjected = preprocessing.scale(y1)
    origEnVecsProjected = np.column_stack((tmp[:,:1],origEnVecsProjected.astype(np.str)))
    origForeignVecsProjected = preprocessing.scale(y2)
    origForeignVecsProjected = np.column_stack((tmp2[:, :1], origForeignVecsProjected.astype(np.str)))
    np.savetxt(outputEnFile,origEnVecsProjected,fmt="%s",delimiter=' ')
    np.savetxt(outputForeignFile,origForeignVecsProjected,fmt="%s",delimiter=' ')
    print "work over!"
def Predeal(origFile,newFile):
    file1 = open(origFile, 'rb')
    file2 = open(newFile, 'wb')
    i=0
    if (file1):
        for line in file1:
            if i!=0:
                tmp = line.strip()+"\n"
                file2.write(tmp)
            i+=1
    else:
        print "Failed to open"
    file1.close()
    file2.close()

if __name__ == "__main__":
    Predeal(origForeignVecFile, origForeignVecFileNew)
    Predeal(origEnVecFile, origEnVecFileNew)
    project_vectors(origForeignVecFile, origEnVecFile, subsetEnVecFile, subsetForeignVecFile, outputEnFile,outputForeignFile,20)