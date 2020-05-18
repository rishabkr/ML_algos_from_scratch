import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class Weather:
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
        normalize_list=['Temperature (C)','Humidity','Wind Speed (km/h)','Wind Bearing (degrees)','Visibility (km)','Pressure (millibars)']
        i=0
        for column in  normalize_list:
            test_values[column]=(test_values[column]-train_mean[i])/train_std[i]
        return (test_values)
        
    def normalize_train(self,column_names,dataset):
        train_mean=[]
        train_std=[]
        for column in column_names:
            #print("Column_name: ",column,"Column_mean: ",dataset[column].mean(),"Column_std :",dataset[column].std())
            #dataset[column]=(dataset[column]-dataset[column].min())/(dataset[column].max()-dataset[column].min())
            train_mean.append(dataset[column].mean())
            train_std.append(dataset[column].std())
            dataset[column]=(dataset[column]-dataset[column].mean())/(dataset[column].std())
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
        #print('weights shape',weights.shape)
        weights,bias,cost=self.gradient_descent(weights,train_values,train_labels,alpha,bias,num_iter)
        return weights,bias

    def predict_test(self,test_value):
        return np.dot(test_value,self.theta)+self.bias

    def predict(self,filename):
        test_data=pd.read_csv(filename)
        test_data=test_data.drop(['Daily Summary','Formatted Date'],axis=1)
        self.test_data=self.normalize_test(test_data,self.train_mean,self.train_std)
        #self.test_labels=self.test_data['Apparent Temperature (C)']
        self.test_data=self.test_data.drop(['Apparent Temperature (C)'],axis=1)
        self.dummies3 = pd.get_dummies(self.test_data["Summary"])
        self.dummies4 = pd.get_dummies(self.test_data["Precip Type"],dummy_na=True)
        self.dummies3=self.dummies3.T.reindex(self.lst1).T.fillna(0)
        self.dummies4=self.dummies4.T.reindex(self.lst2).T.fillna(0)
        self.dummy_test=pd.concat([self.dummies3,self.dummies4],axis=1)
        self.test_df=pd.concat([self.test_data,self.dummy_test],axis=1)
        self.test_df=self.test_df.drop(['Summary','Precip Type'],axis=1)
        self.test_values=np.array(self.test_df)
        # s1=set(self.test_df.columns)
        # s2=set(self.train_df.columns)
        # print(s1-s2)
        prediction=[]
        #test_labels=np.array(test_labels)
        #print(self.test_values.shape)
        for i in range(len(self.test_values)):
            pred=self.predict_test(self.test_values[i])
            prediction.append(pred)
        return prediction


    def train(self,filename):
        dataset=pd.read_csv(filename)
        dataset=dataset.drop(['Daily Summary','Formatted Date'],axis=1)
        self.normalize_list=['Temperature (C)','Humidity','Wind Speed (km/h)','Wind Bearing (degrees)','Visibility (km)','Pressure (millibars)']
        self.train_data,self.train_mean,self.train_std=self.normalize_train(self.normalize_list,dataset)
        self.train_labels=self.train_data['Apparent Temperature (C)']
        self.train_data=self.train_data.drop(['Apparent Temperature (C)'],axis=1)
        self.dummies1 = pd.get_dummies(self.train_data["Summary"])
        self.dummies2 = pd.get_dummies(self.train_data["Precip Type"],dummy_na=True)
        self.dummy=pd.concat([self.dummies1,self.dummies2],axis=1)
        self.train_df=pd.concat([self.train_data,self.dummy],axis=1)
        self.train_df=self.train_df.drop(['Summary','Precip Type'],axis=1)
        self.lst1=self.dummies1.columns
        self.lst1=list(self.lst1)
        self.lst2=self.dummies2.columns
        self.lst2=list(self.lst2)
        self.train_values=np.array(self.train_df)        
        self.train_labels=np.array(self.train_labels)
        #print(self.train_values.shape)
        self.theta,self.bias=self.multi_var_linear_regression(self.train_values,self.train_labels,alpha=0.005,num_iter=2000)
        #print(self.theta.shape)



