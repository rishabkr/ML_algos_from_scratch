from tqdm import tqdm
import numpy as np
from collections import defaultdict
import operator


from itertools import combinations,chain

class Apriori:
    def __init__(self,min_support,min_confidence):
        self.min_support=min_support
        self.min_confidence=min_confidence
        self.order=None
        self.transactions=None
        self.num_transactions=None
        self.Candidates={}
        self.Itemsets={}
        self.Itemset_support_counts={}
        self.Discarded_itemsets={}
        self.order=None
        self.frequent_itemsets={}
        self.itemset_supports=[]
        self.freq_itemsets=[]

    
    def count_itemset_occurance(self,itemsets,transactions):
        count=0
        for i in range(len(transactions)):
            if set(itemsets).issubset(set(transactions[i])):
                count+=1
        return count
    
    def merge(self,itemset1,itemset2):
        itemset1.sort(key=lambda val : self.order.index(val))
        itemset2.sort(key=lambda val : self.order.index(val))
    
        for i in range(len(itemset1)-1):
            if itemset1[i]!=itemset2[i]:
                return []

        if self.order.index(itemset1[-1]) < self.order.index(itemset2[-1]):
            return itemset1+[itemset2[-1]]
        return []
    
    def join_itemsets(self,set_of_itemsets):
        Candidates=[]
        for i in range(len(set_of_itemsets)):
            for j in range(i+1,len(set_of_itemsets)):
                itemsets_out=self.merge(set_of_itemsets[i],set_of_itemsets[j])
                if len(itemsets_out)>0:
                    Candidates.append(itemsets_out)
        return Candidates
    
    def get_freuqent_itemsets(self,itemsets,transactions,minimum_support,previously_discarded):
        supp_count=[]
        new_discarded=[]
        itemsets_list=list()
        num_transactions=len(transactions)

        k=len(previously_discarded)

        for s in tqdm(range(len(itemsets))):
            discarded_before=False

            if(k > 0):
                for item in previously_discarded[k]:
                    if(set(item).issubset(set(itemsets[s]))):
                        discarded_before=True
                        break

            if not discarded_before:
                count=self.count_itemset_occurance(itemsets[s],transactions)
                current_support=count/num_transactions
                if current_support >=minimum_support:
                    itemsets_list.append(itemsets[s])
                    supp_count.append(count)
                else:
                    new_discarded.append(itemsets[s])

        return itemsets_list,supp_count,new_discarded
    
    def get_order(self,transactions):
        mx=-1
        for t in transactions:
            if mx < max(t):
                mx=max(t)
        return mx
    
    def print_table(self,T,supp_count):
        print('Itemset | Frequency')
        for k in range(len(T)):
            print(f'{T[k]}  :  {supp_count[k]}')
        print('\n\n')
    
    def update_tables(self,itemset_size,frequency,support_count,new_discarded): 
        self.Discarded_itemsets.update({itemset_size:new_discarded})
        self.Itemsets.update({itemset_size:frequency})
        self.Itemset_support_counts.update({itemset_size:support_count})

    
    def init(self,transactions):
        self.order=[i for i in range(1,self.get_order(transactions)+1)]
        itemset_size=1
        
        self.Discarded_itemsets={itemset_size:[]}
        self.Candidates.update({itemset_size : [[f] for f in self.order]})
        
        frequency,support_count,new_discarded=self.get_freuqent_itemsets(self.Candidates[itemset_size],transactions,self.min_support,self.Discarded_itemsets)    
        
        self.update_tables(itemset_size,frequency,support_count,new_discarded)
        
        
    def powerset(self,s):
        return list(chain.from_iterable(combinations(s,r) for r in range(1,len(s)+1)))
    
    def generate_itemset_frequency(self,verbose):
        k=2
        convergence=False

        while convergence==False:
            self.Candidates.update({k : self.join_itemsets(self.Itemsets[k-1])})
            
            if(verbose==1):
                print(f'Table Candidates {k} \n ')
                self.print_table(self.Candidates[k],[self.count_itemset_occurance(it,self.transactions) for it in self.Candidates[k]])
            
            frequency,support_count,new_discarded=self.get_freuqent_itemsets(self.Candidates[k],self.transactions,self.min_support,self.Discarded_itemsets)
            
            self.update_tables(k,frequency,support_count,new_discarded)

            if len(self.Itemsets[k])==0:
                convergence=True
            else:
                print(f'Table Itemsets {k} \n ')
                self.print_table(self.Itemsets[k],self.Itemset_support_counts[k])

            k+=1
    
    def create_rule_set(self,itemset,rule_rhs,rule_lhs,confidence,itemset_support):
        self.freq_itemsets.append(str(itemset))
        self.itemset_supports.append(float(itemset_support/self.num_transactions))
    
    
    def generate_association_rules(self):
        association_rules=""
        for i in range(1,len(self.Itemsets)):
            for j in range(len(self.Itemsets[i])):
                s = self.powerset(set(self.Itemsets[i][j]))
                
                s.pop()#remove subset with all the items
                for z in s:
                    S = set(z)
                    X=set(self.Itemsets[i][j])
                    X_S=set(X-S)

                    support_x=self.count_itemset_occurance(X,self.transactions)
                    support_x_s=self.count_itemset_occurance(X_S,self.transactions)

                    confidence=support_x / support_x_s

                    if support_x >= self.min_support and confidence >= self.min_confidence:
                        self.create_rule_set(X,X_S,S,confidence,support_x)

    
    def fit_transform(self,transactions,verbose=0):
        self.transactions=transactions
        self.num_transactions=len(self.transactions)    
        
        self.init(self.transactions)
        
        if(verbose==1):
            self.print_table(self.Itemsets[1],self.Itemset_support_counts[1])
    
        
        self.generate_itemset_frequency(verbose)
        
        self.generate_association_rules()
        
        self.frequent_itemsets={'Itemset':self.freq_itemsets,'Support':self.itemset_supports}
        return pd.DataFrame(self.frequent_itemsets).drop_duplicates()




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

ap=Apriori(min_support=0.05,min_confidence=0.05)

results=ap.fit_transform(transactions2,verbose=1)

print(results)
