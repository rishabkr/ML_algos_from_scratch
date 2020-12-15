import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import operator

class improved_apriori:
    def __init__(self):
        self.one_item_frequency=defaultdict(int)
        self.candidate_itemset_pairs=defaultdict(int)
        self.candidate_itemset_triples=defaultdict(int)
        
        self.frequent_one_itemsets=defaultdict(int)
        self.frequent_itemset_pairs=defaultdict(int)
        self.frequent_itemset_triples=defaultdict(int)
        
        self.min_support=None
        self.min_support_frequency=None
        self.transactions=None
        
        self.final_frequent_itemsets=defaultdict(int)
    
    def merge_itemsets(self,*itemsets):
        return str(sorted(itemsets))
    
    def create_pairsets(self,*itemsets):
        pairs=[]
        for index1 in range(len(itemsets)-1):
            for index2 in range(index1+1,len(itemsets)):
                pairs.append(self.merge_itemsets(itemsets[index1],itemsets[index2]))
        return pairs
    
    def find_one_frequent_candidates(self):
        for transaction in self.transactions:
            for item in transaction:
                self.one_item_frequency[item]+=1

    def find_one_frequent_itemsets(self):
        for item in self.one_item_frequency.keys():
            if(self.one_item_frequency[item] >= self.min_support_frequency):
                self.frequent_one_itemsets[item]=self.one_item_frequency[item]
    
    def find_two_frequent_candidates(self):
        for transaction in tqdm(self.transactions):
            for index1 in range(len(transaction)-1):
                if(transaction[index1] not in self.frequent_one_itemsets):
                    continue
                for index2 in range(index1+1,len(transaction)):
                    if(transaction[index2] not in self.frequent_one_itemsets):
                        continue 
                    candidate_pair=self.merge_itemsets(transaction[index1],transaction[index2])
                    self.candidate_itemset_pairs[candidate_pair]+=1
                    
    def find_two_frequent_itemsets(self):
        for item_pair in self.candidate_itemset_pairs.keys():
            if(self.candidate_itemset_pairs[item_pair] > self.min_support_frequency):
                self.frequent_itemset_pairs[item_pair]=self.candidate_itemset_pairs[item_pair]
    
    def find_three_frequent_candidates(self):
        for transaction in tqdm(self.transactions):
            for index1 in range(len(transaction)-2):
                if(transaction[index1] not in self.frequent_one_itemsets):
                    continue
                for index2 in range(index1+1,len(transaction)-1):
                    if(transaction[index2] not in self.frequent_one_itemsets):
                        continue 
                    pair1=self.merge_itemsets(transaction[index1],transaction[index2])
                    if pair1 not in self.frequent_itemset_pairs:
                        continue

                    for index3 in range(index2+1,len(transaction)):
                        if(transaction[index3] not in self.frequent_one_itemsets):
                            continue

                        all_pairs=self.create_pairsets(transaction[index1],
                                                 transaction[index2],
                                                 transaction[index3])
                        for pair in all_pairs:
                            if(pair not in self.frequent_itemset_pairs):
                                continue
                                
                        itemset_triple=self.merge_itemsets(transaction[index1],
                                                 transaction[index2],
                                                 transaction[index3])

                        self.candidate_itemset_triples[itemset_triple]+=1


    def find_three_frequent_itemsets(self):
        for itemset in self.candidate_itemset_triples.keys():
            if(self.candidate_itemset_triples[itemset] > self.min_support_frequency):
                self.frequent_itemset_triples[itemset]=self.candidate_itemset_triples[itemset]
                
    def find_final_frequent_itemsets(self):
        
        for itemset,freq in self.frequent_one_itemsets.items():
            self.final_frequent_itemsets[itemset]=freq

        for itemset,freq in self.frequent_itemset_pairs.items():
            self.final_frequent_itemsets[itemset]=freq

        for itemset,freq in self.frequent_itemset_triples.items():
            self.final_frequent_itemsets[itemset]=freq
            
        final=sorted(self.final_frequent_itemsets.items(),key=operator.itemgetter(1))
        return final
        
    def fit_transform(self,transaction,min_support=0.1):
        self.transactions=transactions
        self.min_support=min_support
        self.min_support_frequency=self.min_support*len(self.transactions)
        
        self.find_one_frequent_candidates()
        self.find_one_frequent_itemsets()
        
        self.find_two_frequent_candidates()
        self.find_two_frequent_itemsets()
        
        self.find_three_frequent_candidates()
        self.find_three_frequent_itemsets()
        
        final_results=self.find_final_frequent_itemsets()
        
        final_results=np.array(final_results)
        
        final_dict={'Itemset': final_results[:,0], 'Frequency' : final_results[:,1].astype('int32')}
        
        final_df=pd.DataFrame(final_dict)
        return final_df
       


#Simple apriori for k=3
def load_transaction(data_path):
        transactions=[]
        with open(data_path,'r') as data_file:
            for lines in tqdm(data_file):
                transactions_list=list(lines.strip().split())
                transactions_list=[int(x) for x in transactions_list]

                transactions_list=list(np.unique(transactions_list))
                transactions_list.sort()
                transactions.append(transactions_list)

        return transactions

transactions=load_transaction('retail_data.txt')

iap=improved_apriori()

results=iap.fit_transform(transactions,0.05)
print(results)
