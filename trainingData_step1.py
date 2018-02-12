# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#
#import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import numpy as np
#from scipy import stats

import os
#from os import listdir
from os.path import isfile
import docx2txt
from string import punctuation
#import sys
#import nltk
from nltk import word_tokenize as wt
from nltk import FreqDist as fd
from nltk.corpus import stopwords

from datetime import datetime
import pypyodbc
import gc
from sklearn.cross_validation import train_test_split as tts
from rmgarbage.is_garbage import is_garbage
from nltk.stem import SnowballStemmer
import re
import json
#from statsmodels.stats.outliers_influence  import variance_inflation_factor as vif
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:17:22 2017

@author: karl.wong
"""
#sourcepath=r"C:\Users\karl.Wong\Box Sync\GIM_Share_HK\HK Leases\TA\Kowloon Lease Data\20170531\TA"
#targetFolder=r"C:\Users\karl.wong\Box Sync\GIM_Share_HK\HK Leases\TA\Kowloon Lease Data\20151031\TA"
destFolder=r"\\hkpac001\groups$\InfoMgmt\Documents\Karl_Wong\TA-backup\stat_result"
source=[os.path.join(os.environ['USERPROFILE'],r"Box Sync\GIM_Share_HK\HK Leases\TA\Kowloon Lease Data")
,r"\\hkpac001\groups$\\InfoMgmt\Secured\HK Office Leases\_Staging"]
snow = SnowballStemmer('english')
#source=[os.path.join(os.environ['USERPROFILE'],r"Box Sync\GIM_Share_HK\HK Leases\TA\Kowloon Lease Data")]



MaxWordL=4
MaxRankp=0


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        

def returnSQLtoList(query):
    ServerName="USLILVMDWSTG1.delphiprd.am.joneslanglasalle.com"
    MSQLDatabase="GDIM_Discovery"
    conn =pypyodbc.connect("DRIVER={{SQL Server}};SERVER={0}; database={1}; \
                      trusted_connection=yes".format(ServerName,MSQLDatabase))
    cursor = conn.cursor()
    cursor.execute(query)
    cursor.close
    conn.close
    return(np.array(list(cursor.fetchall())).tolist())

def textToWordList(tk,WordLen):
    try:    
        n=len(tk)-WordLen
        lt=[]
        for k in range(n):
            s=""
            for l in range(WordLen):
                s=s+" "+ tk[k+l]
            s=s.strip()
            lt.append(s)
        return(lt)
    except ValueError as e:
        print(e)

def fristXpq(sDat,max_len,x=0):
    token=wt(sDat.lower())
    lt=[]
    if len(sDat)<max_len:
        max_len=len(sDat)
    for i in range(1,max_len+1):
        text=[]
        if i>1:
            text=textToWordList(token,i)
        else:
            text=token
        if x==0:
            fdist=fd(text).most_common()
        else:
            fdist=fd(text).most_common(x)
        fdist=[[fdist[k][0],fdist[k][1]] for tp,k in zip(fdist,range(len(fdist)))]
#        sfdist=[[fdist[k][0],fdist[k][1]/len(text)] for tp,k in zip(fdist,range(len(fdist)))]
        lt.extend(fdist)
#        lt=fqtopq(lt,len(text))
    return(lt)
#def fqtopq(fq,n):
#    return([0 if n==0 else r[1]/n for r in page for page in fq])

sql=("select [file name] "
"from [hk_ta].[staging_src_excel_ta] "
"where isTA={0}")


TAsql=sql.format(1)
nonTAsql=sql.format(0)

TAList=returnSQLtoList(TAsql)
nonTAList=returnSQLtoList(nonTAsql)

k=0
for elm in TAList:
    TAList[k]=''.join(elm)
    k+=1
k=0  
for elm in nonTAList:
    nonTAList[k]=''.join(elm)
    k+=1
    
def inTA(TAdoc):
    if TAdoc in TAList:
        t=1
    else:
        if TAdoc in nonTAList:
            t=0
        else:
            t=-1
    return(t)


#def dataList(lt,sDat):
#    try:
#        for i in range(MaxWordL):
#            tk=textToWordList(wt(sDat),i+1)
#            toAdd=list(set(tk)-set(lt[i]))
#            lt[i].extend(toAdd)
#    except ValueError as e:
#        print(e)
        
def onlyornoVowel(s):
    vowel=[]
    vowel.extend('aeiouy')
    sl=[]
    sl.extend(s)
    overlapset=set(vowel).intersection(set(sl))
    if overlapset==set() or overlapset==set(sl):
        return True
    else:
        return False




def textexclusion(d):
    ex=set(punctuation)
    stops=set(stopwords.words('english'))
    try:
        dt=''
        for i in d.strip().lower().split():
            for j in list(ex):
                i=i.replace(j,'')
            for j in range(10):
                i=i.replace(str(j),'')
            for m in re.findall('[^a-zA-Z0-9]',i):
                i=i.replace(m,'')
            if not is_garbage(i) and i not in (stops or ex)  and not onlyornoVowel(i):
                dt=dt + ' ' + snow.stem(i)     
        return(dt.strip())
    except ValueError as e:
        print(e)

def countTA(mainLt,compareLt):
    return len(set(compareLt).intersection(set(mainLt)))

#tar=r'C:\Users\karl.wong\Box Sync\GIM_Share_HK\HK Leases\TA\Kowloon Lease Data\20170630\TA'



TAprec={}
nonTAprec={}
X_TA=[]
X_nonTA=[]
maxSamplesize=500
for sourcePath in source:
    for subFolder in os.listdir(sourcePath):
        if (not isfile(os.path.join(sourcePath,subFolder))) and (subFolder.isnumeric()):
            if subFolder>'20160101':
                for subsubFolder in os.listdir(os.path.join(sourcePath,subFolder)):
                    tar=os.path.join(sourcePath,subFolder,subsubFolder)
                    if os.path.exists(tar) and os.path.isdir(tar):
                        if subsubFolder.find('TA')>-1:
                            TA_List=[f for f in os.listdir(tar) if f.endswith('.docx')]
                            for TA_docx in TA_List:
                                wordpath=os.path.join(tar,TA_docx)
                                if TA_docx[0]!="~":
                                    if len(textexclusion(docx2txt.process(wordpath).lower()))>0:
                                        TA_docxS=TA_docx.replace('.docx','')
                                        if inTA(TA_docxS)==1:
                                            X_TA.append(TA_docxS)
                                        else:
                                            if inTA(TA_docxS)==0:
                                                X_nonTA.append(TA_docxS)
                        else:
                            if subsubFolder.find('NoValue')>-1:
                                TA_List=[f for f in os.listdir(tar) if f.endswith('.docx')]
                                for TA_docx in TA_List:
                                    wordpath=os.path.join(tar,TA_docx)
                                    if TA_docx[0]!="~":
                                        if len(textexclusion(docx2txt.process(wordpath).lower()))>0:
                                            TA_docxS=TA_docx.replace('.docx','')
                                            if inTA(TA_docxS)==0:
                                                X_nonTA.append(TA_docx.replace('.docx',''))

def sampleSelect(X):
    if len(X) >maxSamplesize:
        X_train,Xtest=tts(X,test_size=1-maxSamplesize/len(X))
        return(X_train)
    else:
        return(X)
TAsample=sampleSelect(X_TA)
nonTAsample=sampleSelect(X_nonTA)
for sourcePath in source:      
    for subFolder in os.listdir(sourcePath):
        if (not isfile(os.path.join(sourcePath,subFolder)))  and (subFolder.isnumeric()):
            if subFolder>'20160101':
                for subsubFolder in os.listdir(os.path.join(sourcePath,subFolder)):
                    tar=os.path.join(sourcePath,subFolder,subsubFolder)
                    if os.path.exists(tar) and os.path.isdir(tar):
                        if subsubFolder.find('TA')>-1 or subsubFolder.find('NoValue')>-1:
                            TA_List=[f for f in os.listdir(tar) if f.endswith('.docx')]
                            print(tar)
                            print(datetime.now())
                            for TA_docx in TA_List:
                                TA_docxS=TA_docx.replace('.docx','')
                                if TA_docx[0]!="~" and inTA(TA_docxS)!=-1:
                                    if TA_docxS in TAsample or TA_docxS in nonTAsample:
                                        try:
                                            wordpath=os.path.join(tar,TA_docx)
                                            dat_ex=textexclusion(docx2txt.process(wordpath).lower())
                                            if dat_ex !='':
                                                plen=len(list(set(wt(dat_ex))))
                                                pq=fristXpq(dat_ex,MaxWordL,int(plen*MaxRankp))
                                                pqDat=df([list(df(pq)[1])],columns=df(pq)[0],index=[TA_docxS])
                                                
                                                if inTA(TA_docxS)==1:
                                                    try:
                                                        TAprec[os.path.join(subFolder,subsubFolder)]=TAprec[os.path.join(subFolder,subsubFolder)].append(pqDat)
                                                    except:
                                                        TAprec[os.path.join(subFolder,subsubFolder)]=pqDat
                                                else:
                                                    try:
                                                        nonTAprec[os.path.join(subFolder,subsubFolder)]=nonTAprec[os.path.join(subFolder,subsubFolder)].append(pqDat)
                                                    except:
                                                        nonTAprec[os.path.join(subFolder,subsubFolder)]=pqDat
                                                del pq
                                                del pqDat
                                                del dat_ex
                                                del wordpath
                                                del plen
                                        except ValueError as e:
                                            print(e)
                                            print(os.path.join(tar,TA_docxS))
                                            TA_List.remove(TA_docxS)
                                            pass


#
                  
                        


def dic_to_DF(dic):
    f=df()
    for d in dic:
        f=pd.concat([f,dic[d]])
    return(f)


def dicDF_to_dicJson(dic):
    dicjson={}
    for c in dic:
        if dic[c].shape!=(0,0):
            dicjson[c]=dic[c].to_json()
    return(dicjson)

TAprec_json=dicDF_to_dicJson(TAprec)
nonTAprec_json=dicDF_to_dicJson(nonTAprec)


minCount=2
def dicDFchoice(dic):
    s=pd.Series()
    for c in dic:
        s=s.add(dic[c].count(),fill_value=0)
    return list(s.loc[s>minCount].index)

TAdocumentchoice=dicDFchoice(TAprec_json)
nonTAdocumentchoice=dicDFchoice(nonTAprec_json)



jsonPath=r'N:\InfoMgmt\Documents\Karl_Wong\TA-backup\stat_result'
ensure_dir(os.path.join(jsonPath,'TA'))
ensure_dir(os.path.join(jsonPath,'nonTA'))






for c in TAprec_json:
    with open(os.path.join(jsonPath,'TA\{0}.json'.format(c.replace('\\',''))), 'w') as fp:
        json.dump(TAprec_json[c][list(set(TAdocumentchoice).intersaction(set(TAprec_json[c])))], fp)
  
for c in nonTAprec_json:
    with open(os.path.join(jsonPath,'nonTA\{0}.json'.format(c.replace('\\',''))), 'w') as fp:
        json.dump(nonTAprec_json[c][list(set(nonTAdocumentchoice).intersaction(set(nonTAprec_json[c])))], fp)        
#
#
#TA_prec_df=dic_to_DF(TAprec)
#nonTA_prec_df=dic_to_DF(nonTAprec)
#
#
##def calTF(df):
##    return((1+np.log(df))*np.log(max(len(nonTAsample),len(TAsample))/df.count()))
##TA_prec_df=calTF(TA_prec_df)
##nonTA_prec_df=calTF(nonTA_prec_df)
#
#TAcountDF=TA_prec_df.count()
#TAcountDF=TAcountDF.loc[TAcountDF>2]
#nonTAcountDF=nonTA_prec_df.count()
#nonTAcountDF=nonTAcountDF.loc[nonTAcountDF>2]
#TAword=list(TAcountDF.to_frame().index)
#nonTAword=list(nonTAcountDF.to_frame().index)
#
#
#
##OverLapWord=list(set(TAword).intersection(set(nonTAword)))
##TAonlyword=list(set(TAword)-set(OverLapWord))
##nonTAonlyword=list(set(nonTAword)-set(OverLapWord))
##TAonlyword=df(TAonlyword)
##nonTAonlyword=df(nonTAonlyword)
##OverLapcountDF=TAcountDF[OverLapWord]+nonTAcountDF[OverLapWord]
##TA_prec_df=TA_prec_df[TAword]
##nonTA_prec_df=nonTA_prec_df[nonTAword]
##
##del TAword
##del nonTAword
#
##gc.collect()
##
##
##alpha=0.05
##alphaU=0.05
##def normalizeVector(X):
##    lenVector=df([[r for _ in range(len(X.columns))] for r in np.sqrt((X.transpose()*X.transpose()).sum())],columns=X.columns,index=X.index)
##    return(X/lenVector)
##
##normalizeTA= normalizeVector(TA_prec_df)
##normalizenonTA= normalizeVector(nonTA_prec_df)
##del TA_prec_df
##del nonTA_prec_df
#
#
#
##def pvOfMannWhitneyUtest(X,Y):
##    return(manu(X,Y,use_continuity=False,alternative='two-sided').pvalue)
##
##OverlapWord=df([r for r in OverLapWord if pvOfMannWhitneyUtest(normalizeTA[r].fillna(0),normalizenonTA[r].fillna(0))<=alphaU])
##
##normalizeOverlap=pd.concat([normalizeTA[OverlapWord[0]],normalizenonTA[OverlapWord[0]]],axis=0)
##
##
##gc.collect()
#
#
#
#csvPath = os.path.join(destFolder,'trainingdata_without_featureselection_{0}.csv')
#
#TA_prec_df[TAword].to_csv(csvPath.format('TA'),encoding='utf8')
#del TA_prec_df
#nonTA_prec_df[nonTAword].to_csv(csvPath.format('nonTA'),encoding='utf8')
#del nonTA_prec_df
#gc.collect()

#normalizeTA.to_csv(csvPath.format('TAonly'),encoding='utf8')
#del normalizeTA
#normalizenonTA.to_csv(csvPath.format('nonTAonly'),encoding='utf8')
#del normalizenonTA
#normalizeOverlap.to_csv(csvPath.format('Overlap'),encoding='utf8')
#del normalizeOverlap

