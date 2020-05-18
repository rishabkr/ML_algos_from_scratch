
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math


class DecisionTree:
    def __init__(self):
        pass
    #for regression
    def calculate_mse(self,data):
        if len(data[:,-1])==0:
          mse=0
        else:
          #data[:-,1] is the last column which contains the labels of the price/value/label  
          mse= np.mean((data[:,-1]-np.mean(data[:,-1]))**2)
        return mse

#determine type of feature
    def continuous_or_categorical(self,df):
        #to see the number of unique values in each type of feature
        conti_or_categ=[]
        for column in df.columns:
            if df.dtypes[column] == "object":
                conti_or_categ.append("categorical")
            else:
                conti_or_categ.append("continuous")    
        return conti_or_categ


    def split_data(self,data,split_column,split_value):
    #now insetead of comparing features,we first find out if it is continuous or categorical
    #if continuous we can compare with <= or >= if categorical we use ==
        split_column_values=data[:,split_column]
        type_of_feature=TYPES_OF_FEATURES[split_column]
        if type_of_feature=="continuous":
            data_less_than_split=data[split_column_values<=split_value]
            data_more_than_split=data[split_column_values>split_value]
        else:
            data_less_than_split=data[split_column_values==split_value]
            data_more_than_split=data[split_column_values!=split_value]
        return data_less_than_split,data_more_than_split

    #we pass dataframe so that we dont need to pass train or test after converting to a 2-d array
     #this function takes min_samples to check if there are sufficient data points in that side of the tree
     #if less than min_samples,we can classify the data even if  the data is not pure yet
     #this way the height of the tree gets reduced and unncessary calculations/splits are avoided
     #min_samples default value=2
     #this is called PRUNING of the tree it also checks the max depth which we want in our tree
     #FOR REGRESSION
    def opt_decision_tree_algorithm(self,df,counter=0,min_samples=7,max_depth=5,first_iteration=True):
        #when counter=0 we change it to 2-d array else we leave it as it is
        #as other functions use 2-d array
        #data preperation
        if (first_iteration):
            data=df.values
        else:
            data=df #called as 2-d array
    
        #basecase 
        if ((len(data)<min_samples) or (counter==max_depth)):
            return np.mean(data[:,-1])

        else:
            counter+=1
            first_iteration=False
            split_positions={}
            #will contain list of potentila  split values for that feature/column
            rows,num_of_columns=data.shape
            #exclude the label column
            for curr_column in range(num_of_columns-1):
                values=data[:,curr_column]
                unique_values=np.unique(values)
                split_positions[curr_column]=unique_values
 
       
        #finding the split which gives the least mean square error
        best_mse_lst =list()
        for column_index in split_positions:
            for value in split_positions[column_index]:
                data_less_than,data_more_than=self.split_data(data,column_index,value)
                n_data_points=len(data_less_than)+len(data_more_than)
                    #weights of both types of data(weighted mse)
                current_mse=((len(data_less_than)/n_data_points)*self.calculate_mse(data_less_than)+(len(data_more_than)/n_data_points)*self.calculate_mse(data_more_than))
                best_mse_lst.append((current_mse, column_index, value))

        best_mse, split_column, split_value = min(best_mse_lst)
        data_less_than,data_more_than=self.split_data(data,split_column,split_value)
        
        #checkforemptyclass
        if len(data_less_than)==0 or len(data_more_than)==0:
            return np.mean(data[:-1])
        
        #instantiate subtree (recurse) tells the column name where split
        #to get only the inndex use split_column in .format()
        feature_name=COLUMN_NAMES[split_column]
        type_of_feature=TYPES_OF_FEATURES[split_column]
        if type_of_feature=="continuous":
            criteria="{} <= {}".format(feature_name,split_value)
        else:
            criteria="{} == {}".format(feature_name,split_value)
        
        tree_node={criteria: []}
        
        #find_subtree_answers
        left_subtree_condn=self.opt_decision_tree_algorithm(data_less_than,counter,min_samples,max_depth,first_iteration)
        right_subtree_condn=self.opt_decision_tree_algorithm(data_more_than,counter,min_samples,max_depth,first_iteration)
        
        #append only the name if not splitting
        if left_subtree_condn==right_subtree_condn:
            tree_node=right_subtree_condn
        else:
            tree_node[criteria].append(left_subtree_condn)
            tree_node[criteria].append(right_subtree_condn)

        return tree_node

    def train(self, train_path):
        df=pd.read_csv(train_path)
        #df.info()

        df=df.drop(["Id","Alley","PoolQC","MiscFeature","Fence","FireplaceQu"],axis=1)
        df=df.rename(columns={"SalePrice":"label"})
        df["MSZoning"].replace('C (all)', 'C', inplace=True)
        #nalist=df.columns[df.isna().any()].tolist()

        mean_frontage=df.LotFrontage.mean()
        mode_MasVnrType=df.MasVnrType.mode()[0]
        mean_MasVnrArea=df.MasVnrArea.mean()
        mode_BsmtQual=df.BsmtQual.mode()[0]
        mode_BsmtCond=df.BsmtCond.mode()[0]
        mode_BsmtExposure=df.BsmtExposure.mode()[0]
        mode_BsmtFinType1=df.BsmtFinType1.mode()[0]
        mode_BsmtFinType2=df.BsmtFinType2.mode()[0]
        mode_Electrical=df.Electrical.mode()[0]
        mode_GarageType=df.GarageType.mode()[0]
        mean_GarageYrBlt=df.GarageYrBlt.mean()
        mode_GarageFinish=df.GarageFinish.mode()[0]
        mode_GarageQual=df.GarageQual.mode()[0]
        mode_GarageCond=df.GarageCond.mode()[0]

        df=df.fillna({"LotFrontage":mean_frontage,"MasVnrType":mode_MasVnrType,"MasVnrArea":mean_MasVnrArea,"BsmtQual":mode_BsmtQual,"BsmtCond":mode_BsmtCond,"BsmtExposure":mode_BsmtExposure,"BsmtFinType1":mode_BsmtFinType1,"BsmtFinType2":mode_BsmtFinType2,"Electrical":mode_Electrical,"GarageType":mode_GarageType,"GarageYrBlt":mean_GarageYrBlt,"GarageFinish":mode_GarageFinish,"GarageQual":mode_GarageQual,"GarageCond":mode_GarageCond})
        
        best_max_depth = 7
        best_min_samples = 10
        global COLUMN_NAMES
        global TYPES_OF_FEATURES
        TYPES_OF_FEATURES=self.continuous_or_categorical(df)
        COLUMN_NAMES=df.columns        
        self.current_decision_tree=self.opt_decision_tree_algorithm(df, max_depth=best_max_depth, min_samples=best_min_samples,first_iteration=True)

    def get_decision(self,decision_tree,criteria,value,operator,feature,df_row):
        if(operator=="<="):
            value=float(value)
                    #ask question
            if df_row[feature]<=value:
                trueorfalse=decision_tree[criteria][0]
            else:
                trueorfalse=decision_tree[criteria][1]
                        #the value can be a integer or a string
        elif(operator!="<="):
            if str(df_row[feature])==value:
                trueorfalse=decision_tree[criteria][0]
            else:
                trueorfalse=decision_tree[criteria][1]
        return trueorfalse        
     #to classify the data to classes(get the class name)
    def predict_example(self,df_row,decision_tree):
         #get the splitting_criteria(key)
        criteria=list(decision_tree.keys())[0]
        #get the answers(elements of the string)
        feature,operator,value=criteria.split(" ")
        trueorfalse=self.get_decision(decision_tree,criteria,value,operator,feature,df_row)
        #basecase if answer is not a dictionary
        if not isinstance(trueorfalse,dict):
            return trueorfalse
        else:
            sub_tree=trueorfalse
            return self.predict_example(df_row,sub_tree)

    def predict(self,test_path):
        df_test=pd.read_csv(test_path)
        #df_labels=pd.read_csv("test_labels.csv")
        #df_test=df_test.join(df_labels,lsuffix='_test', rsuffix='_labels')

        df_test=df_test.drop(["Id","Alley","PoolQC","MiscFeature","Fence","FireplaceQu"],axis=1)

        #nalist=df_test.columns[df_test.isna().any()].tolist()

        tmean_frontage=df_test.LotFrontage.mean()
        tmode_MasVnrType=df_test.MasVnrType.mode()[0]
        tmean_MasVnrArea=df_test.MasVnrArea.mean()
        tmode_BsmtQual=df_test.BsmtQual.mode()[0]
        tmode_BsmtCond=df_test.BsmtCond.mode()[0]
        tmode_BsmtExposure=df_test.BsmtExposure.mode()[0]
        tmode_BsmtFinType1=df_test.BsmtFinType1.mode()[0]
        tmode_BsmtFinType2=df_test.BsmtFinType2.mode()[0]
        tmode_GarageType=df_test.GarageType.mode()[0]
        tmean_GarageYrBlt=df_test.GarageYrBlt.mean()
        tmode_GarageFinish=df_test.GarageFinish.mode()[0]
        tmode_GarageQual=df_test.GarageQual.mode()[0]
        tmode_GarageCond=df_test.GarageCond.mode()[0]


        df_test=df_test.fillna({"LotFrontage":tmean_frontage,"MasVnrType":tmode_MasVnrType,"MasVnrArea":tmean_MasVnrArea,"BsmtQual":tmode_BsmtQual,"BsmtCond":tmode_BsmtCond,"BsmtExposure":tmode_BsmtExposure,"BsmtFinType1":tmode_BsmtFinType1,"BsmtFinType2":tmode_BsmtFinType2,"GarageType":tmode_GarageType,"GarageYrBlt":tmean_GarageYrBlt,"GarageFinish":tmode_GarageFinish,"GarageQual":tmode_GarageQual,"GarageCond":tmode_GarageCond})
        df_test["MSZoning"].replace('C (all)', 'C', inplace=True)
        predictions = df_test.apply(self.predict_example, args=(self.current_decision_tree,), axis=1)
        
        return list(predictions)

    
 
if __name__ == '__main__':
    dt = DecisionTree()
    dt.train("train.columnssv")
    preds = dt.predict("test.csv")
    #print("Done Testing")
    df_labels=pd.read_csv("test_labels.csv")
    df_labels = df_labels.iloc[:, 1]
    #print(len(preds), df_labels.shape)
    label_vals=df_labels.to_numpy()
    #print (r2_score(label_vals, preds))