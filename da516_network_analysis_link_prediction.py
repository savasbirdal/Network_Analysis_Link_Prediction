# -*- coding: utf-8 -*-
"""
Created on Mon May 14 18:16:33 2018

@author: sbirdal
"""

import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite
from networkx.algorithms import community
from networkx.algorithms import connectivity
import itertools
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import GridSearchCV
from keras.optimizers import RMSprop
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from networkx.algorithms import connectivity
from sklearn.naive_bayes import GaussianNB
schools=pd.read_csv("Schools.csv")
projects=pd.read_csv("Projects.csv")
projects=projects.merge(schools, how="left",left_on="School ID",right_on="School ID")
projects=projects[projects["School State"]=="California"]
projectlist=projects["Project ID"].values.tolist()
donors=pd.read_csv("Donors.csv")
#donorlist=donors["Donor ID"].values.tolist()
donations=pd.read_csv("Donations.csv")
donations = donations[donations["Project ID"].isin(projectlist)]
donationslist=donations["Donor ID"].values.tolist()
donors=donors[donors["Donor ID"].isin(donationslist)]
#donations=donations[~donations["Project ID"].isin(donorlist)]
donations=donations.groupby(["Donor ID","Project ID"]).agg({'Donation Received Date': np.max, 'Donation Amount': np.sum})
donations=donations.reset_index()
donations=donations.sort_values("Donation Received Date")
donations_train=donations.iloc[:10000,:]
donations_test=donations.iloc[10000:12000,:]
projects=projects[projects["Project ID"].isin(donations_train["Project ID"].values.tolist())]
donors=donors[donors["Donor ID"].isin(donations_train["Donor ID"].values.tolist())]
#donG=nx.from_pandas_edgelist(donations_train,source="Donor ID",target="Project ID",edge_attr=True,create_using=nx.DiGraph(),)
donB=nx.DiGraph()
donB.add_nodes_from(donors["Donor ID"].values.tolist(),project=0)
donB.add_nodes_from(projects["Project ID"].values.tolist(),project=1)
donB.add_weighted_edges_from(donations_train[["Donor ID","Project ID","Donation Amount"]].values.tolist())
projectlabel=nx.get_node_attributes(donB,"project")
donornodes={n for n, d in donB.nodes(data=True) if d['project']==0}
donorG = bipartite.projected_graph(donB, donornodes)
#remove = [node for node,degree in list(donortodonor.degree()) if degree < 3]
#donortodonor.remove_nodes_from(remove)
wccs=list(nx.weakly_connected_component_subgraphs(donB))
sortedwc=sorted(wccs,key=lambda x:len(x.nodes()),reverse=True)
largestwcc=sortedwc[0]
len(donB.nodes())
len(list(largestwcc.nodes()))
comdf=pd.DataFrame()
for i in range(len(communities)):
    print(len(communities[i]))
start=time.time()
community_generator=community.girvan_newman(largestwcc)
for i in range(29):
    communities=next(community_generator)
    print("number of comm:"+str(len(communities)))
    for j in range(len(communities)):
        print("size:"+str(len(donB.subgraph(communities[j]).nodes())))
with open('communities30.txt', 'w') as filehandle:  
    for listitem in list(communities):
        filehandle.write('%s\n' % listitem)
communities=[]
with open('communities.txt', 'r') as file:  
    for line in file:
        content=line.rstrip("\n")
        content=content.rstrip("}")
        content=content.lstrip("{")
        content=list(content.split(","))
        final=[]
        for i in content:
            j=i.strip()
            final.append(j.strip("'"))
        communities.append(final)
       
end=time.time()
print(start-end)

from collections import defaultdict
####################ENDS#
nodedict=defaultdict()
simrankmatrix,nodes_i=simrank(donB)
sortednodes=sorted(nodes_i,key=nodes_i.__getitem__,reverse=False)
for node in donB.subgraph(communities[0]).nodes():  
    index=nodes_i[node]
    recomdict=defaultdict(int)
    for recomnode in donB.subgraph(communities[0]).nodes():       
        recomdict[recomnode]=nx.jaccard_coefficient(donB.subgraph(communities[0]),node,recomnode)
    recomsort=sorted(recomdict,key=recomdict.__getitem__,reverse=True)
    nodedict[node]=recomsort

simrankmatrix[nodes_i[communities[3][0]]][nodes_i[nodedict[communities[3][0]][0]]]
x_train=[]
y_train=[]
count=0
for edge in list(donB.subgraph(communities[14]).edges()):
    com_neigh=len(list(nx.common_neighbors(donB.subgraph(communities[14]).to_undirected(),edge[0],edge[1])))
    conn=connectivity.node_connectivity(donB.subgraph(communities[14]).to_undirected(),edge[0],edge[1])
    pref_att=list(nx.preferential_attachment(donB.subgraph(communities[14]).to_undirected(),[edge]))[0][2]
    #jac_coef=list(nx.jaccard_coefficient(donB.to_undirected(),[edge]))[0][2]
    x_train.append([com_neigh,pref_att,conn])
    y_train.append(1)
    count+=1
    print(count/len(list(donB.subgraph(communities[14]).edges())))
x_train2=x_train.copy()
y_train2=y_train.copy()
count=0
non_edges=list(nx.non_edges(donB.subgraph(communities[14])))[:6550]
for edge in non_edges:
    com_neigh=len(list(nx.common_neighbors(donB.subgraph(communities[14]).to_undirected(),edge[0],edge[1])))
    conn=connectivity.node_connectivity(donB.subgraph(communities[14]).to_undirected(),edge[0],edge[1])       
    pref_att=list(nx.preferential_attachment(donB.subgraph(communities[14]).to_undirected(),[edge]))[0][2]
    #sjac_coef=list(nx.jaccard_coefficient(donB.subgraph(communities[0]).to_undirected(),[edge]))[0][2]
    x_train2.append([com_neigh,pref_att,conn])
    y_train2.append(0)
    count+=1
    print(count/len(non_edges))

nbclassifier = MultinomialNB()
print ("10 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(nbclassifier, x_train2,y_train2,cv=10 , scoring='accuracy')))
#SVM K-fold with cross_val_score
svmclassifier = SVC(random_state = 0 , kernel='sigmoid',gamma=1.0)
print ("10 Fold CV Score for SVM: ",np.mean(cross_val_score(svmclassifier, x_train2,y_train2,cv=10 , scoring='precision')))

for i in range(5000):
    for j in range(5000):
        if((simrankmatrix[i][j]>0)&(simrankmatrix[i][j]<1)):
            print(simrankmatrix[i][j])
            print(str(i)+str(j))
            break

donationscheckdonor=donations_train["Donor ID"].values.tolist()
donationscheckproject=donations_train["Project ID"].values.tolist()
donationscheck=donationscheckproject+donationscheckdonor
donations_test=donations_test[donations_test["Donor ID"].isin(communities[14])]
donations_test=donations_test[donations_test["Project ID"].isin(communities[14])]
x_test=[]
y_test=[]
for index,row in donations_test.iterrows():
    com_neigh=len(list(nx.common_neighbors(donB.subgraph(communities[14]).to_undirected(),row["Donor ID"],row["Project ID"])))
    pref_att=list(nx.preferential_attachment(donB.subgraph(communities[14]).to_undirected(),[(row["Donor ID"],row["Project ID"])]))[0][2]
    conn=connectivity.node_connectivity(donB.subgraph(communities[14]).to_undirected(),row["Donor ID"],row["Project ID"])
    #jac_coef=list(nx.jaccard_coefficient(donB.to_undirected(),[edge]))[0][2]
    x_test.append([com_neigh,pref_att,conn])
    y_test.append(1)
non_edges2=list(nx.non_edges(donB.subgraph(communities[14])))[6551:6661]
for edge in non_edges2:
    com_neigh=len(list(nx.common_neighbors(donB.subgraph(communities[14]).to_undirected(),edge[0],edge[1])))
    pref_att=list(nx.preferential_attachment(donB.subgraph(communities[14]).to_undirected(),[edge]))[0][2]
    conn=connectivity.node_connectivity(donB.subgraph(communities[14]).to_undirected(),edge[0],edge[1])
    #sjac_coef=list(nx.jaccard_coefficient(donB.subgraph(communities[0]).to_undirected(),[edge]))[0][2]
    x_test.append([com_neigh,pref_att,conn])
    y_test.append(0)
    count+=1
    print(count/len(non_edges))
from sklearn.utils import shuffle
x_train2,y_train2=shuffle(x_train2,y_train2,random_state=0)
nbclassifier.fit(x_train2,y_train2)

y_pred=nbclassifier.predict(x_test)
score=metrics.precision_score(y_test,y_pred)
print(score)
svmclassifier=SVC(random_state = 0 , kernel='sigmoid',gamma=1.0)
svmclassifier.fit(x_train2,y_train2)
svm_pred=svmclassifier.predict(x_test)
score=metrics.accuracy_score(y_test,svm_pred)
print(score)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train2, y_train2)
print ("10 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(clf, x_train2,y_train2,cv=10 , scoring='precision')))
#clf.score(x_validation_tfidf, y_validation)
clf.score(x_train2, y_train2)
clf.score(x_test, y_test)
y_pred=clf.predict(x_test)
metrics.precision_score(y_test,y_pred)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import GridSearchCV
from keras.optimizers import RMSprop
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
#1024,512,128
x_train_keras=np.matrix(x_train2).reshape(7205,3)
def make_model():
    model_n = Sequential()
    model_n.add(Dense(128,activation='relu',input_dim=7205))
    optimizer=RMSprop()
    #model_n.add(Dense(1000,activation='tanh'))
    #model_n.summary()
    model_n.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])
    #model_n.add(Dense(1500, activation='relu', input_dim=x_train_tfidf.shape[1]))
    #model_n.add(Dense(100, activation='tanh'))
    #custom_adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #model_n.add(Dense(1, activation='sigmoid'))
    #model_n.add(Dropout(0.05))
    #model_n.compile(optimizer=custom_adam,
    #              loss='binary_crossentropy',
    #              metrics=['accuracy'])
    return model_n
#model_n=make_model()
#model_n.fit(x_train_tfidf, y_train, batch_size=32, epochs=3, verbose=1 )
#model_n.evaluate(x_test_tfidf,y_test,batch_size=32,verbose=1)
seed = 7
np.random.seed(seed)
model_n=make_model()
epoch_no=3
batch_no=32
neural_network = KerasClassifier(make_model, batch_size=batch_no,epochs=epoch_no)
cv_scores=cross_val_score(estimator=neural_network,X=x_train_keras,y=np.matrix(y_train2).reshape(1,7205),cv=10)
np.average(cv_scores)


model_n.fit(x_train_cvec, y_train, batch_size=batch_no, epochs=epoch_no,verbose=1)
model_n.evaluate(x_test_cvec,y_test,batch_size=batch_no,verbose=1)
nx.write_weighted_edgelist(donB,"edgelist.txt",delimiter=',')
nx.read_edgelist()