# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 00:02:14 2020

@author: Andrei-Laptop
"""
import numpy as np
import random
from sklearn import svm

red=open('winequality-red.csv','r')
white=open('winequality-white.csv','r')
# CITIREA DATELOR
AtributeVinRosu=[]
EticheteVinRosu=[]

AtributeVinAlb=[]
EticheteVinAlb=[]

for rand in red:
    atribute=rand.split(';')
    EticheteVinRosu.append(atribute[-1]) #ultimul element este defapt Calitatea care trebuie determinata
    atribute.pop(-1) #o scoatem dupa memorare
    AtributeVinRosu.append(atribute) #si adaugam celelalte atribute
AtributeVinRosu.pop(0)#deoarece contine denumirile
EticheteVinRosu.pop(0)

for rand in white: #facem acelasi lucru si pentru vinul alb
    atribute=rand.split(';')
    EticheteVinAlb.append(atribute[-1])
    atribute.pop(-1)
    AtributeVinAlb.append(atribute)
AtributeVinAlb.pop(0)
EticheteVinAlb.pop(0)


# SEPARAREA DATELOR: date de antrenare si de testare
# alegem 10% din date, la intamplare pentru testare
AtributeVinRosuTest=[]
EticheteVinRosuTest=[]
for i in range(int(np.ceil(0.1*len(AtributeVinRosu)))):
    x=random.randrange(0,len(AtributeVinRosu)-i)
    AtributeVinRosuTest.append(AtributeVinRosu[x])
    AtributeVinRosu.pop(x)
    EticheteVinRosuTest.append(EticheteVinRosu[x])
    EticheteVinRosu.pop(x)
    #luam atributele si etichetelor vinurilor alese random
    
    
AtributeVinAlbTest=[]
EticheteVinAlbTest=[] #pentru vinul alb luam 25% pt test, deoarece avem mai multe instante
for i in range(int(np.ceil(0.25*len(AtributeVinAlb)))):
    x=random.randrange(0,len(AtributeVinAlb)-i)
    AtributeVinAlbTest.append(AtributeVinAlb[x])
    AtributeVinAlb.pop(x)
    EticheteVinAlbTest.append(EticheteVinAlb[x])
    EticheteVinAlb.pop(x)    
    
#transformam totul in array-uri numpy pentru a fi siguri ca functioneaza cu sklearn
AtributeVinRosu=np.array(AtributeVinRosu)
AtributeVinRosuTest=np.array(AtributeVinRosuTest)
EticheteVinRosu=np.array(EticheteVinRosu)
EticheteVinRosuTest=np.array(EticheteVinRosuTest)

AtributeVinAlb=np.array(AtributeVinAlb)
AtributeVinAlbTest=np.array(AtributeVinAlbTest)
EticheteVinAlb=np.array(EticheteVinAlb)
EticheteVinAlbTest=np.array(EticheteVinAlbTest)

# ANTRENAREA DATELOR

clasificatorRosu=svm.SVC(C=0.1,kernel='poly',degree=5).fit(AtributeVinRosu,EticheteVinRosu)
ClasificareRosu=clasificatorRosu.predict(AtributeVinRosuTest)

clasificatorAlb=svm.SVC(C=0.1,kernel='poly',degree=5).fit(AtributeVinAlb,EticheteVinAlb)
ClasificareAlb=clasificatorAlb.predict(AtributeVinAlbTest)

# VERIFICAM ACURATETEA
PreziseBine=0
for i in range(len(EticheteVinRosuTest)):
    Apropiate=[int(EticheteVinRosuTest[i]),int(EticheteVinRosuTest[i])-1,int(EticheteVinRosuTest[i])+1]
    #considerand ca valorile x-1,x,x+1 sunt apropiate si reprezinta o nota similara
    if int(ClasificareRosu[i]) in Apropiate:
        PreziseBine+=1
print('Pentru vinul rosu avem un procent de: '+str(format(PreziseBine/len(EticheteVinRosuTest)*100,'.2f'))+'%')

PreziseBine=0 # si pentru vinul alb
for i in range(len(EticheteVinAlbTest)):
    Apropiate=[int(EticheteVinAlbTest[i]),int(EticheteVinAlbTest[i])-1,int(EticheteVinAlbTest[i])+1]
    if int(ClasificareAlb[i]) in Apropiate:
        PreziseBine+=1
print('Pentru vinul alb avem un procent de: '+str(format(PreziseBine/len(EticheteVinAlbTest)*100,'.2f'))+'%')