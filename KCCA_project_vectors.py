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
from PyKCCA.kernels import GaussianKernel
from PyKCCA.kernels import DiagGaussianKernel
from PyKCCA.kernels import PolyKernel

datadir = "/home/xfbai/mywork/git/KCCA-Experiment/data/"
OutputDir="/home/xfbai/mywork/git/KCCA-Experiment/Output/"
#origForeignVecFile = "/home/xfbai/tmpvec/new_embedding_size200.fr"
origForeignVecFile = "/home/xfbai/tmpvec/new_embedding_size200.zh"
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
    tmp3 = np.loadtxt(subsetEnVecFile, dtype=np.str, delimiter=' ')
    subsetEnVecs = tmp3[:, 1:].astype(np.float)
    tmp4 = np.loadtxt(subsetForeignVecFile,dtype=np.str,delimiter=' ')
    subsetForeignVecs = tmp4[:, 1:].astype(np.float)

    '''预处理，使每行正则化'''
    origEnVecs=preprocessing.scale(origEnVecs)
    origForeignVecs=preprocessing.scale(origForeignVecs)
    #subsetEnVecs = preprocessing.scale(subsetEnVecs)
    #subsetForeignVecs = preprocessing.scale(subsetForeignVecs)
  
    '''训练CCA'''
    x1 = subsetEnVecs
    x2 = subsetForeignVecs
    resDict={}
    '''paramSet=[(18,10),(19,4),(19,6),(20,1),(20,5),(20,7),(20,8),(20,9),(21,2),(21,4	),(21,6),(21,9),(22,6),(22,7),(23,6),(23,8),(24,2),(24,5),(24,6),(24,7),(24,8)		,(24,9)]
    for i ,j in paramSet:
    	for k in range(1,10,2):
	    kernel1 = GaussianKernel(float(i))
            kernel2 = GaussianKernel(float(j))
    #kernel = DiagGaussianKernel()
    	    cca = KCCA(kernel1, kernel2,
               		regularization=k/10.0,
               		decomp='full',
               		method='kettering_method',
               		scaler1=lambda x: x,
               		scaler2=lambda x: x,
	       		SSnum=NUMCC).fit(x1, x2)
    	    print "sigma1=",i,"sigma2=",j,"regularization=",k/10.0,cca.beta_
    	    y1, y2 = cca.transform(origEnVecs, origForeignVecs)
    	    origEnVecsProjected = preprocessing.scale(y1)
    	    origEnVecsProjected = np.column_stack((tmp[:,:1],origEnVecsProjected.astype(np.str)))
    	    origForeignVecsProjected = preprocessing.scale(y2)
    	    origForeignVecsProjected = np.column_stack((tmp2[:, :1], origForeignVecsProjected.astype(np.str)))
    	    np.savetxt(outputEnFile,origEnVecsProjected,fmt="%s",delimiter=' ')
    	    np.savetxt(outputForeignFile,origForeignVecsProjected,fmt="%s",delimiter=' ')
	    a,b  =   commands.getstatusoutput( 'python /home/xfbai/mywork/git/KCCA-Experiment/qvec/qvec_cca2.py')
	    print b
	    ff = open(paramFile,'a')
	    ttmpstr = str((i,j,k))+b+"\n"
	    ff.write(ttmpstr)
   	    ff.close()
     '''	
    for i in range(3,7):
	for j in range(3,7):
		for k in range(1,7,1):
    			kcca =KCCA('rbf','rbf',regularization=10**-k,gamma1=10**-i,gamma2=10**-j,n_jobs=-1,n_components=NUMCC).fit(x1,x2)
    			# print kcca.corrs
    			y1, y2 = kcca.transform(origEnVecs, origForeignVecs)
    			origEnVecsProjected = preprocessing.scale(y1)
    			origEnVecsProjected = np.column_stack((tmp[:,:1],origEnVecsProjected.astype(np.str)))
    			origForeignVecsProjected = preprocessing.scale(y2)
    			origForeignVecsProjected = np.column_stack((tmp2[:, :1], origForeignVecsProjected.astype(np.str)))
    			np.savetxt(outputEnFile,origEnVecsProjected,fmt="%s",delimiter=' ')
   	 		np.savetxt(outputForeignFile,origForeignVecsProjected,fmt="%s",delimiter=' ')
    			a,b  =   commands.getstatusoutput( 'python /home/xfbai/mywork/git/KCCA-Experiment/evalution/qvec/qvec_cca2.py')
    			#c,d = commands.getstatusoutput('bash run.sh evaluateBiDict fr /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_en_out.txt /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_foreign_out.txt')
			c,d = commands.getstatusoutput('bash run.sh evaluateBiDict zh /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_en_out.txt /home/xfbai/mywork/git/KCCA-Experiment/Output/KCCA_foreign_out.txt')
    			print b
    			print d    			
			ff = open(paramFile,'a')
    			ttmpstr = str((10**-i,10**-j,10**-k))+' '+b+d+"\n"
    			ff.write(ttmpstr)
    			ff.close()	

    print "Work Finished"

if __name__ == "__main__":
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print "training model..."
    project_vectors(origForeignVecFile, origEnVecFile, subsetEnVecFile, subsetForeignVecFile, outputEnFile,outputForeignFile,100)
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
