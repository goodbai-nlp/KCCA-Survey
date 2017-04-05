#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: KCCAtest2.py
@time: 17-3-31 下午12:20
"""

#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: KCCA_project_Vector.py
@time: 17-3-31 上午11:13
"""
import numpy as np
import time
from sklearn import preprocessing
# from kcca import KernelCCA
# from kernel.GaussianKernel import GaussianKernel
from PyKCCA.kcca import KCCA
from PyKCCA.kernels import GaussianKernel
from PyKCCA.kernels import DiagGaussianKernel
from PyKCCA.kernels import PolyKernel

datadir = "/home/jackbai/mywork/Git/KCCA-Experiment/data/"
OutputDir="/home/jackbai/mywork/Git/KCCA-Experiment/Output/"
# origForeignVecFile = "/home/xfbai/tmpvec/embeddings_size20.fr"
# origForeignVecFileNew = "/home/xfbai/tmpvec/new_embeddings.fr"
# origEnVecFile = "/home/xfbai/tmpvec/embeddings_size20.en"
# origEnVecFileNew = "/home/xfbai/tmpvec/new_embeddings.en"
# subsetEnVecFile = datadir+"Out_en_new_aligned.txt"
# subsetForeignVecFile = datadir+"Out_foreign_new_aligned.txt"

# datadir = "/home/jackbai/PycharmProjects/KCCAtest2/data/"
# OutputDir="/home/jackbai/PycharmProjects/KCCAtest2/Output/"
origForeignVecFile = datadir+"de-sample.txt"
origForeignVecFileNew = datadir+"new_embeddings.fr"
origEnVecFile = datadir+"en-sample.txt"
origEnVecFileNew = datadir+"new_embeddings.en"
subsetEnVecFile = datadir+"en_new_aligned.txt"
subsetForeignVecFile = datadir+"de_new_aligned.txt"

outputEnFile = OutputDir+"KCCA_en_out.txt"
outputForeignFile = OutputDir+"KCCA_foreign_out.txt"

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
    # kernel = DiagGaussianKernel()
    # kcca = KernelCCA(kernel,kernel,0.1)
    cca = KCCA(kernel, kernel,
               regularization=1e-5,
               decomp='icd',
               method='simplified_hardoon_method',
               scaler1=lambda x: x,
               scaler2=lambda x: x,
               SSnum=NUMCC).fit(x1, x2)
    print cca.beta_
    y1, y2 = cca.transform(origEnVecs, origForeignVecs)
    # kcca.learnModel(x1,x2)
    # print kcca.lmbdas
    # y1,y2=kcca.project(origEnVecs,origForeignVecs)
    origEnVecsProjected = preprocessing.scale(y1)
    origEnVecsProjected = np.column_stack((tmp[:,:1],origEnVecsProjected.astype(np.str)))
    origForeignVecsProjected = preprocessing.scale(y2)
    origForeignVecsProjected = np.column_stack((tmp2[:, :1], origForeignVecsProjected.astype(np.str)))
    np.savetxt(outputEnFile,origEnVecsProjected,fmt="%s",delimiter=' ')
    np.savetxt(outputForeignFile,origForeignVecsProjected,fmt="%s",delimiter=' ')
    print "Work Finished"
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
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    Predeal(origForeignVecFile, origForeignVecFileNew)
    Predeal(origEnVecFile, origEnVecFileNew)
    print "predeal complete!"
    print "training model..."
    project_vectors(origForeignVecFileNew, origEnVecFileNew, subsetEnVecFile, subsetForeignVecFile, outputEnFile,outputForeignFile,20)
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))