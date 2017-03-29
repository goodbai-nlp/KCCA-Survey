#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: project_vectors.py
@time: 17-3-29 上午9:28
"""
import numpy as np
from sklearn import preprocessing
import rcca

datadir = "/home/jackbai/mywork/Git/KCCA-Experiment/data/"
OutputDir="/home/jackbai/mywork/Git/KCCA-Experiment/Output/"
origForeignVecFile = datadir+"de-sample.txt"
origEnVecFile = datadir+"en-sample.txt"
subsetEnVecFile = datadir+"en_new_aligned.txt"
subsetForeignVecFile = datadir+"de_new_aligned.txt"
outputEnFile = OutputDir+"en_out.txt"
outputForeignFile = OutputDir+"foreign_out.txt"

def project_vectors(origForeignVecFile,origEnVecFile,subsetEnVecFile,subsetForeignVecFile,outputEnFile,outputForeignFile,truncRatio):
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
    cca = rcca.CCA(kernelcca=False, reg=0., numCC=20)
    cca.train([subsetEnVecs, subsetForeignVecs])

    '''生成投影后的向量'''
    tmpOutput = rcca._listdot([d.T for d in [subsetEnVecs, subsetForeignVecs]], cca.ws)
    origEnVecsProjected = preprocessing.scale(tmpOutput[0])
    origForeignVecsProjected = preprocessing.scale(tmpOutput[1])
    np.savetxt(outputEnFile,origEnVecsProjected,fmt="%.5f",delimiter=' ')
    np.savetxt(outputForeignFile,origForeignVecsProjected,fmt="%.5f",delimiter=' ')
    print "work over!"

if __name__ == "__main__":
    project_vectors(origForeignVecFile, origEnVecFile, subsetEnVecFile, subsetForeignVecFile, outputEnFile,outputForeignFile, 0.5)