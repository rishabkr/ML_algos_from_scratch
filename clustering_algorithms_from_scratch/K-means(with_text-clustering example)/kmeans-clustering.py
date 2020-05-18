import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy import spatial
import os
from collections import Counter
import re
import string

np.random.seed(1000000)

class Cluster:
    def distance(self,p1,p2):
    #print(1 - spatial.distance.cosine(p1,p2))
    #return 1 - spatial.distance.cosine(p1,p2)
        return np.sqrt(np.sum((p1-p2)**2))


    def closest_centroid(self,sample):
        distances=[]
        for point in self.centroids:
            d=self.distance(sample,point)
            distances.append(d)
        closest=np.argmin(distances)
        #print(closest)
        return closest

    def create_clusters(self,X_train,K):
        clusters=[[] for _ in range(K)]
        for index,sample in enumerate(X_train):
            centroid_index=self.closest_centroid(sample)
            clusters[centroid_index].append(index)
        return clusters

    def get_centroids(self,clusters,K,num_features,X_train):
        self.centroids=np.zeros((K,num_features))
        for cluster_index,cluster in enumerate(clusters):
            cluster_mean=np.mean(X_train[cluster],axis=0)
            self.centroids[cluster_index]=cluster_mean
        return self.centroids    

    def cleanpunc(self,sentence):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in sentence if ch not in exclude)


    def kmeans(self,K,X_train,num_iterations,plot_steps=False):
        self.clusters=[[] for _ in range(K)]
        self.centroids=[]
        num_samples=X_train.shape[0]
        num_features=X_train.shape[1]
        cnt=0
        #initializing centroids between 0 and number of samples
        random_indexes=np.random.choice(num_samples,K,replace=False)
        #random_indexes=[191, 183, 84, 797, 1695]
        #random_indexes=kmeanspp(X_train,K)
        self.centroids=[X_train[index] for index in random_indexes]
        #print(random_indexes)
        while(cnt<=num_iterations):
            cnt+=1
            #print(cnt)
            self.clusters=self.create_clusters(X_train,K)
            self.centroids_old=self.centroids
            self.centroids=self.get_centroids(self.clusters,K,num_features,X_train)
            #centroids=kmeanspp(X_train,K)
            #if(cnt%10==0):    
                #print(cnt)
            completed=[]
            for i in range(K):
                d=self.distance(self.centroids_old[i],self.centroids[i])
                completed.append(d)
            #print(sum(completed))
            if(sum(completed)==0):
                break
        #return get_cluster_labels(clusters,num_samples)
        return self.clusters
    
    def cluster(self,dirname):
        file_names=os.listdir(dirname)
        #print(file_names)
        text_data={'files':list(), 'data':list()}
        for filename in file_names:
            with open(dirname+'\\'+filename,"r") as data:
                 if filename in text_data:
                     continue
                 text_data['files'].append(filename)
                 text_data['data'].append(data.read())
        
        labels=[]
        for name in file_names:
            label=name.split('_')
            label=label[1].split('.')
            labels.append(label[0])

        text_data['labels']=labels
        df=pd.DataFrame(text_data)
        #print(df)
        i=0
        list_of_sentences=[]
        for sent in df['data'].values:
            filtered_sentence=[]
            for w in sent.split():
                for cleaned_words in self.cleanpunc(w).split():
                    if(cleaned_words.isalpha()):    
                        filtered_sentence.append(cleaned_words.lower())
                    else:
                        continue 
            list_of_sentences.append(filtered_sentence)
            df.loc[i,'filtered_data']=' '.join(filtered_sentence)
            i+=1
            #print(i)
            

        actual=df['labels'].values
        #df['filtered_data']=df['filtered_data'].astype(str)
        #print(df['filtered_data'].iloc[25])
        from sklearn.feature_extraction.text import TfidfVectorizer
        Tfidf_vect = TfidfVectorizer(stop_words='english')
        Train_X_Tfidf=Tfidf_vect.fit_transform(df['filtered_data'])
        Train_X_Tfidf=np.array(Train_X_Tfidf.todense())

        clusters=self.kmeans(5,Train_X_Tfidf,200,plot_steps=False)
        klst=[1,2,3,4,5]
        #print(clusters)
        cluster_labels=[]
        for k in klst:
           ans = Counter(actual[clusters[k-1]]).most_common()
           cluster_labels.append(ans[0][0])
        final_labels=[0 for _ in range(len(actual))]
        for i in range(len(clusters)):
            for j in clusters[i]:
                final_labels[j] = cluster_labels[i]      
        prediction={'files':list(), 'predicted':list()}
        prediction['files']=df['files']
        prediction['predicted']=final_labels
        return (prediction)