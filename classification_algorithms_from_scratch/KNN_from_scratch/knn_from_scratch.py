import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import random
import math
from collections import Counter

class KNNClassifier:
    def __init__(self):
        pass
    def train_test_split(self, dataframe,test_size):
        dataframe_size=len(dataframe)
        if isinstance(test_size,float):#if test size is passed as a proportion
            test_size=round(test_size*dataframe_size)
        #pick random samples from the data for train test split
        indexes=dataframe.index.tolist()
        test_indices=random.sample(population=indexes,k=test_size)
        #now putting the values of train and test data into the respective df's
        test_dataframe=dataframe.loc[test_indices]
        cropped_dataframe=dataframe.drop(test_indices)
        train_dataframe=cropped_dataframe
        return train_dataframe,test_dataframe
    
    def minkowski(self,test_value,p):
        if(p==2):
            distance=np.sum((self.train_values - test_value)**2,axis=1)
            return distance
        elif(p==1):
            distance=np.sum(abs(self.train_values - test_value),axis=1)
            return distance


    def KNeighbors(self, k, test_value,p=2):
        neighbors=[]
        train_length=self.train_values.shape[0]
        if(p==2):
            distance=self.minkowski(test_value,p=2)
        elif(p==1):
            distance=self.minkowski(test_value,p=1)
        k_neighbors=np.argsort(distance)
        k_neighbors=k_neighbors[:k]
        return k_neighbors

    def find_majority(self, k_index):
        ans = Counter(self.train_labels[k_index]).most_common()
        return ans[0][0]

    def train(self, train_path):
        df=pd.read_csv(train_path,header=None)
        
        letters={"a":int(ord('a')),"b":int(ord('b')),"c":int(ord('c')),"d":int(ord('d')),"e":int(ord('e')),"f":int(ord('f')),"g":int(ord('g')),"h":int(ord('h'))
                                             ,"i":int(ord('i')),"j":int(ord('j')),"k":int(ord('k')),"l":int(ord('l')),"m":int(ord('m')),"n":int(ord('n')),"o":int(ord('o')),"p":int(ord('p')),"q":int(ord('q')),"r":int(ord('r'))
                                             ,"s":int(ord('s')),"t":int(ord('t')),"u":int(ord('u')),"v":int(ord('v')),"w":int(ord('w')),"x":int(ord('x')),"y":int(ord('y')),"z":int(ord('z'))} 	         
        for column in df.columns:
            df[column]=df[column].replace(to_replace ="?",value =df[column].mode()[0])
        for column in df.columns:
            df[column]=df[column].replace(to_replace=letters)
		
        df = df.apply(pd.to_numeric)    

        train_df,val_df=self.train_test_split(df,0.3)
		


        train_digits=train_df.to_numpy()
        train_digits=np.array(train_digits)
        val_digits=val_df.to_numpy()
        val_digits=np.array(val_digits)

        self.train_values=train_digits[:,1:]
        self.train_labels=train_digits[:,0]
        self.val_values=val_digits[:,1:]
        self.val_labels=val_digits[:,0]
    
    def predict(self, test_path):
        df_test=pd.read_csv(test_path,header=None)
        letters={"a":int(ord('a')),"b":int(ord('b')),"c":int(ord('c')),"d":int(ord('d')),"e":int(ord('e')),"f":int(ord('f')),"g":int(ord('g')),"h":int(ord('h'))
		                                             ,"i":int(ord('i')),"j":int(ord('j')),"k":int(ord('k')),"l":int(ord('l')),"m":int(ord('m')),"n":int(ord('n')),"o":int(ord('o')),"p":int(ord('p')),"q":int(ord('q')),"r":int(ord('r'))
		              	                               ,"s":int(ord('s')),"t":int(ord('t')),"u":int(ord('u')),"v":int(ord('v')),"w":int(ord('w')),"x":int(ord('x')),"y":int(ord('y')),"z":int(ord('z'))} 
        for column in df_test.columns:
            df_test[column]=df_test[column].replace(to_replace ="?",value =df_test[column].mode()[0])
        for column in df_test.columns:
            df_test[column]=df_test[column].replace(to_replace=letters)


        test_vals=df_test.to_numpy()
        test_vals=np.array(test_vals)
        prediction=[]
        length=test_vals.shape[0]
        for i in range(length):
            k_index=self.KNeighbors(5,test_vals[i])
            result=self.find_majority(k_index)
            prediction.append(result)
        predictions=[]
        for i in range(0,len(prediction)):
    	    predictions.append(chr(prediction[i]))
        return predictions


if __name__ == '__main__':
    knn = KNNClassifier()
    knn.train("train.csv")
    preds = knn.predict("test.csv")
    print("Done Testing")
    df_labels=pd.read_csv("test_labels.csv", header=None)
    label_vals=df_labels.iloc[:, 0].to_numpy()
    label_vals=np.array(label_vals)
    #print(preds.shape)
    #print(label_vals.shape)
    preds=np.array(preds)
    acc = np.sum(preds == label_vals)/preds.shape[0]
    print(acc)

# df_test_labels=pd.read_csv("test_labels.csv",header=None)


# # In[254]:


# for column in df_test_labels.columns:
#     df_test_labels[column]=df_test_labels[column].replace(to_replace=letters)


# # In[255]:


# test_vals=df_test.to_numpy()
# test_vals=np.array(test_vals)
# label_vals=df_test_labels.to_numpy()
# label_vals=np.array(label_vals)


# # In[256]:


# def KNeighbors(k,train_values,train_labels,test_value):
#     neighbors=[]
#     train_length=train_values.shape[0]
#     distance=np.sum((train_values - test_value)**2,axis=1)
#     k_neighbors=np.argsort(distance)
#     k_neighbors=k_neighbors[:k]
#     return k_neighbors


# # In[257]:


# def find_majority(k_index,train_labels):
#     from collections import Counter
#     ans = Counter(train_labels[k_index]).most_common()
#     return ans[0][0]


# # In[258]:


# predictions=[]
# length=test_vals.shape[0]
# for i in range(length):
#     k_index=KNeighbors(6,train_values,train_labels,test_vals[i])
#     result=find_majority(k_index,train_labels)
#     predictions.append(result)
# predictions


# # In[ ]:





# # In[259]:


# cnt=0
# for i in range(0,length):
#     if(predictions[i]==label_vals[i]):
#         cnt+=1
# print(cnt/label_vals.shape[0])


# # In[260]:


# from sklearn.neighbors import KNeighborsClassifier
# classifier=KNeighborsClassifier(n_neighbors=6,metric='minkowski',p=2)
# classifier.fit(train_values,train_labels)


# # In[261]:


# y_pred=classifier.predict(test_vals)


# # In[262]:


# cnt=0
# for i in range(0,length):
#     if(y_pred[i]==label_vals[i]):
#         cnt+=1
# print(cnt/label_vals.shape[0])


# # In[281]:


# from sklearn.metrics import confusion_matrix
# y_pred=y_pred.reshape(1000,1)

