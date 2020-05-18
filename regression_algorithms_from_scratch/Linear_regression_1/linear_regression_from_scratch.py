import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class Airfoil:
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
    
    def normalize_test(self,test_values,train_mean,train_std):
        for i in range(test_values.shape[1]):
            test_values[:,i]=test_values[:,i]-train_mean[i]
            test_values[:,i]=test_values[:,i]/train_std[i]
        return test_values
        
    
    def normalize(self,dataset):
        train_mean=[]
        train_std=[]
        for i in range(dataset.shape[1]):
            mean=np.mean(dataset[:,i])
            train_mean.append(mean)
            std=np.std(dataset[:,i])
            train_std.append(std)
            dataset[:,i]=(dataset[:,i]-mean)/std
        return dataset,train_mean,train_std
        

    def gradient_descent(self,weights,train_values,train_labels,alpha,bias,num_iter):
        num_samples=train_values.shape[0]
        dim=train_values.shape[1]
        cost=np.ones(num_iter)
        i=0
        #print(weights,bias)
        for i in range(num_iter):
            predict=np.dot(train_values,weights)+bias
            cost[i]=(1/(2*num_samples)*sum(np.square(predict-train_labels)))
            #print(cost[i])
            #print(train_values.shape)
            dw=1/(num_samples)*np.dot(train_values.T,(predict-train_labels))
            db=1/(num_samples)*np.sum(predict-train_labels)
            weights-=alpha*dw
            bias-=alpha*db
            i+=1
        return weights,bias,cost


    def multi_var_linear_regression(self,train_values,train_labels,alpha,num_iter):
        train_dimension=train_values.shape[1]
        num_samples=train_values.shape[0]
        ones=np.ones((train_values.shape[0],1))
        bias=0
        weights=np.zeros(train_dimension)
        weights,bias,cost=self.gradient_descent(weights,train_values,train_labels,alpha,bias,num_iter)
        return weights,bias

    def predict_test(self,test_value):
        return np.dot(test_value,self.theta)+self.bias

    def predict(self,filename):
        test_data=pd.read_csv(filename,header=None)
        test_data=np.array(test_data)
        test_data=test_data[:,:-1]
        prediction=[]
        self.test_values=self.normalize_test(test_data,self.train_mean,self.train_std)
        for i in range(len(self.test_values)):
            pred=self.predict_test(self.test_values[i])
            prediction.append(pred)
        return prediction

    def train(self,filename):
        dataset=pd.read_csv(filename,header=None)    	
        train_data=np.array(dataset)
        self.train_values=train_data[:,:-1]
        self.train_labels=train_data[:,-1]
        self.train_values,self.train_mean,self.train_std=self.normalize(self.train_values)
        self.theta,self.bias=self.multi_var_linear_regression(self.train_values,self.train_labels,alpha=0.1,num_iter=20000)
        


