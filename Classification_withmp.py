# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:17:14 2017

@author: Karl.Wong
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:20:26 2017

@author: Karl.Wong
"""

import pandas as pd
import os
from pandas import DataFrame as df
import numpy as np
import gc
import multiprocess as mp
from datetime import datetime

mp_cpu_usage=mp.cpu_count()-1
sourcePath=r'N:\InfoMgmt\Documents\Karl_Wong\TA-backup\stat_result'

def dataProcessing(t):
    import multiprocess as mp
    import os
    jsonPath=os.path.join(sourcePath,t)
    Json_List=[os.path.join(jsonPath,f) for f in os.listdir(os.path.join(jsonPath)) if f.endswith('.json')]
    DF_s=df()
    def json_to_lt(path):
        import json
        import pandas as pd
        with open(path) as data_file:    
            return pd.read_json(json.load(data_file))
    if __name__ == '__main__':
        with mp.Pool(mp_cpu_usage) as p:
            st=datetime.now()
            print(datetime.now())
            trainingData=p.map(json_to_lt, Json_List)
            print(datetime.now())
            print(datetime.now()-st)
            p.close()
        count=pd.Series()
        rcount=pd.Series()
        for d in trainingData:
            count=count.add(d.count(),fill_value=0)
            rcount=rcount.add(d.count(axis=1),fill_value=0)
        
        selection=count.loc[count>np.percentile(count,80)]
        for d in trainingData:
            col=list(set(d.columns).intersection(set(selection.index)))
            DF_s=pd.concat([DF_s,d[col].divide(rcount[d.index],axis=0)])
#            DF_s=pd.concat([DF_s,d[col]])
        del trainingData
        del selection
        del count
        del rcount
        gc.collect()
    return(DF_s)
    

TADF_s=dataProcessing('TA')
nonTADF_s=dataProcessing('nonTA')
X=pd.concat([TADF_s,nonTADF_s])
Overlapcol=list(set(TADF_s.columns).intersection(set(nonTADF_s.columns)))

def doc_term_DF(arg):
    import pandas as pd
    from pandas import DataFrame as df
    import os
    import docx2txt
    from string import punctuation
    #import sys
    #import nltk
    from nltk import word_tokenize as wt
    from nltk import FreqDist as fd
    from nltk.corpus import stopwords
    
    from datetime import datetime
    import gc
    from rmgarbage.is_garbage import is_garbage
    from nltk.stem import SnowballStemmer 
    import re
    MaxWordL=3
    MaxRankp=0
    snow = SnowballStemmer('english')
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
    
    result=df()
    TA_List=[f for f in os.listdir(arg[0]) if f.endswith('.docx')]
    print(arg[0])
    print(datetime.now())
    for TA_docx in TA_List:
        if TA_docx not in arg[1] and TA_docx[0] !="~":
            try:
                wordpath=os.path.join(arg[0],TA_docx)
                dat_ex=textexclusion(docx2txt.process(wordpath).lower())
                if dat_ex!='':
                    plen=len(list(set(wt(dat_ex))))
                    pq=fristXpq(dat_ex,MaxWordL,int(plen*MaxRankp))
                    col=list(set(df(pq)[0]).intersection(set(arg[2])))
                    N=sum(list(df(pq)[1]))
                    pqDat=df([list(df(pq)[1])]/N,columns=list(df(pq)[0]),index=[TA_docx.replace('.docx','')])[col]
                    result=pd.concat([result,pqDat])
                    del pq
                    del pqDat
                    del dat_ex
                    del wordpath
                    del plen
            except ValueError as e:
                print(e)
                print(os.path.join(arg[0],TA_docx))
                TA_List.remove(TA_docx)
                pass
    gc.collect()
    return result


from os.path import isfile
destFolder=r"\\hkpac001\groups$\InfoMgmt\Documents\Karl_Wong\TA-backup\stat_result"
sourcePathList=[r"N:\InfoMgmt\Secured\HK Office Leases\_Staging",os.path.join(os.environ['userprofile'],r"Box Sync\GIM_Share_HK\HK Leases\TA\Kowloon Lease Data")]
#sourcePathList=[os.path.join(os.environ['userprofile'],r"Box Sync\GIM_Share_HK\HK Leases\TA\Kowloon Lease Data")]


pathtoreadl=[ [[os.path.join(sourcePath,subFolder,subsubFolder) 
for subsubFolder in os.listdir(os.path.join(sourcePath,subFolder)) 
    if (subsubFolder=='TA' or subsubFolder=='NoValue') 
    and os.path.join(sourcePath,subFolder,subsubFolder)
    and os.path.exists(os.path.join(sourcePath,subFolder,subsubFolder)) and os.path.isdir(os.path.join(sourcePath,subFolder,subsubFolder))] 
for subFolder in os.listdir(sourcePath) 
    if (not isfile(subFolder)) 
    and (subFolder.isnumeric()) 
    and (int(subFolder) in range(20170601,20170901))] 
for sourcePath in sourcePathList]

    

def readbottomelt(lt,r):
    for sublt in lt:
        if type(sublt)==list:
            readbottomelt(sublt,r)
        else:
            r.append(sublt)

pathtoread=[]
readbottomelt(pathtoreadl,pathtoread)
mp_cpu_usage=mp.cpu_count()

exList=list(set(TADF_s.index)|set(nonTADF_s.index))   
colList=list(set(TADF_s.columns)|set(nonTADF_s.columns)) 
arg=zip(pathtoread,[exList for _ in range(len(pathtoread))],[colList for _ in range(len(pathtoread))])


def tfidf(x):
    if type(x)!=type(df()):
        x=df(x)
    return np.log(1+x.fillna(0))*np.log(x.shape[0]/x.count())
    
   
if __name__ == '__main__':
    with mp.Pool(mp_cpu_usage) as p:
            st=datetime.now()
            print(datetime.now())
            undefinedData=p.map(doc_term_DF,arg)
            print(datetime.now())
            print(datetime.now()-st)
            p.close()
            undefinedDF=df()
    for d in undefinedData:
        undefinedDF=pd.concat([undefinedDF,d])
        X_t=X[list(set(X.columns).intersection(set(undefinedDF.columns)))]
    #    X_tfidf=tfidf(X_t) 
        Y=[1]*TADF_s.shape[0]
        Y.extend([0]*nonTADF_s.shape[0])
def main_(ntopic):
    
    def fitlda(x):
        from sklearn.decomposition import LatentDirichletAllocation as lda
        from pandas import DataFrame as df
    #    l=lda(n_components =100)
        l=lda(n_topics =ntopic)#This parameter has been renamed to n_components and will be removed in version 0.21. .. deprecated:: 0.19
        if type(x)==type(df()):
            x=x.fillna(0)
        l.fit(x.fillna(0))
        return(l)
    if __name__ == '__main__':
        with mp.Pool(mp_cpu_usage) as p:
            clflt=p.map(fitlda,[TADF_s,nonTADF_s,X_t])

        def topicfound(arg):
            import numpy as np
            def vectorNomalize(v):
                return(np.divide(v,np.sqrt(np.sum(np.multiply(v,v)))).tolist())
            topic_words=[]
            topic_weight=[]
            for topic in arg[0].components_:
                word_idx = np.argsort(vectorNomalize(topic))[::-1][:22]
                topic_words.append([arg[1].columns[i] for i in word_idx])
                topic_weight.append([topic[i] for i in word_idx])
            return [topic_words,topic_weight]
        with mp.Pool(mp_cpu_usage) as p:
            xtopic=p.map(topicfound,zip(clflt,[TADF_s,nonTADF_s,X_t]))
            
    
    #
    #    topiclmt=[]
    #    for x in xtopic[2][1]:
    #        mu=np.mean(x)
    #        sigma=np.std(x)
    #        for k in range(len(x)):
    #            if x[k]<mu+1.33*sigma:
    #                break
    #        topiclmt.append(k)
        xlt=[]
        testlt=[]
    
    
        for i in range(len(xtopic[2][0])):
            col=xtopic[2][0][i]#[:topiclmt[i]]
            w=xtopic[2][1][i]#[:topiclmt[i]]
            xlt.append([float(i[0]) for i in np.matrix(X_t[col].fillna(0))*np.transpose(np.matrix(w))])
            testlt.append([float(i[0]) for i in np.matrix(undefinedDF[col].fillna(0))*np.transpose(np.matrix(w))])
    
    #    from mpl_toolkits.mplot3d import Axes3D        
    #    import matplotlib.pyplot as plt
    #
    #    def mpplt(arg):
    #        import matplotlib.pyplot as plt
    #        import numpy as np
    #        plt.figure(arg[0])
    #        
    #        k=1
    #        for Xj in arg[2]:
    #            plt.subplot(arg[4],arg[5],k,aspect='equal')
    #            for x1,x2,y in zip(arg[1],Xj,arg[3]):
    #                x1=np.log(x1)
    #                x2=np.log(x2)
    #                if y==0:
    #                    plt.scatter(x1, x2, c="red",s=1)
    #                else:
    #                    plt.scatter(x1,x2,c="blue",s=1)
    #            k+=1
    #        plt.show()
    #    import math
    #    st=5
    #    m=8
    #    pltarg=[]
    #    for i in range(st,st+m):
    #        X1=xlt[i]   
    #        X2=[]
    #        for j in range(i+1,st+m+1):
    #            X2.append(xlt[j])
    #        n=st+m-i
    #        h=max(1,int(np.sqrt(n)))
    #        pltarg.append([i,X1,X2,Y,h,math.ceil(n/h)])
    #    if __name__ == '__main__':
    #        with mp.Pool(mp_cpu_usage) as p:        
    #            p.map(mpplt,pltarg)
    #    plt.show()
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection='3d')
    #    for x1,x2,x3,y in zip(xlt[0],xlt[1],xlt[2],Y):
    #        x1=np.log(x1)
    #        x2=np.log(x2)
    #        x3=np.log(x3)
    #        if y==0:
    #            ax.scatter(x1,x2,x3, c="red",s=1,depthshade=False)
    #        else:
    #            ax.scatter(x1,x2,x3,c="blue",s=1,depthshade=False)
    #    plt.show()
        def logX(x):
            x=np.abs(np.transpose(np.array((np.log(x)))))
            x[x==np.inf]=0
            return x.tolist()
        Xlda=logX(xlt)
        testlda=logX(testlt)
        def tree_repeat(arg):
            from sklearn import tree
            from pandas import DataFrame as df
            possible=[]
            clf = tree.DecisionTreeClassifier(criterion=arg[3])
            clf.fit(arg[0],arg[1])
            result=clf.predict(df(arg[2]).fillna(0))
            possible.extend([arg[2].index[i] for r,i in zip(result,range(len(result))) if r==1])
            return(possible)
        n=1000
        c='gini'
        with mp.Pool(mp_cpu_usage) as p:
            possibleTA_tree_gini=p.map(tree_repeat,zip([Xlda]*n,[Y]*n,[df(testlda,index=list(undefinedDF.index))]*n,[c]*n))
            c='entropy'
            possibleTA_tree_entropy=p.map(tree_repeat,zip([Xlda]*n,[Y]*n,[df(testlda,index=list(undefinedDF.index))]*n,[c]*n))
        from nltk import FreqDist as fd
        def list_to_fd_DF(lt,t):
            lt_fd=fd(lt)
            id_=[]
            row=[]
            for col in lt_fd:
                row.append([col,lt_fd[col]])
                id_.append(col)
            return(df(row,index=id_,columns=['FileName_{0}'.format(t),'Count_{0}'.format(t)]))
        
        def multilttosinglelt(mlt):
            rlt=[]
            if type(mlt)==list:
                for l in mlt:
                    if type(l)==list:
                        rlt.extend(l)
                    else:
                        rlt.append(l)
            return(rlt)
            
            
        ginilt=multilttosinglelt(possibleTA_tree_gini)
        entropylt=multilttosinglelt(possibleTA_tree_entropy)
            
        possibleTA_tree_gini_DF=list_to_fd_DF(ginilt,'gini')
        possibleTA_tree_entropy_DF=list_to_fd_DF(entropylt,'entropy')
        possibleTA_tree_combined_DF=pd.concat([possibleTA_tree_gini_DF,possibleTA_tree_entropy_DF],axis=1)
        
        def countrequirment(d):
            return(d.loc[d['Count_entropy']==1000].loc[d['Count_gini']==1000])
        destFolder=r"N:\InfoMgmt\Documents\Karl_Wong\TA-backup\stat_result"
        writer = pd.ExcelWriter(os.path.join(destFolder,'tree_new_{0}.xlsx'.format(ntopic)))
        possibleTA_tree_gini_DF.to_excel(writer,'gini')
        possibleTA_tree_entropy_DF.to_excel(writer,'entropys')
        countrequirment(possibleTA_tree_combined_DF).to_excel(writer,'combined')
        writer.save()


main_(100)
