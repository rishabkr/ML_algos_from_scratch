import numpy as np
from collections import defaultdict
from tqdm import tqdm
import operator

class fp_tree_node:
    def __init__(self,item_name,frequency,parent):
        self.parent=parent
        self.item_name=item_name
        self.frequency=frequency
        self.child_node=defaultdict()
        self.next_node=None
        
    def update_item_frequency(self,frequency):
        self.frequency+=frequency
        
    def display_fp_tree(self,index=1):
        print('  '*index + f' Item : {self.item_name}  Frequency: {self.frequency}  Parent : {self.parent}')
        for names,child in self.child_node.items():
            child.display_fp_tree(index+1)


class build_fp_tree:
    def __init__(self,transactions,min_support_count):
        self.transactions=transactions
        self.min_support=min_support_count
        self.header_table=defaultdict()
        self.fp_root=None
    
    def is_frequent(self,frequency):
        if(frequency < self.min_support):
            return False
        else:
            return True
    
    
    def get_frequent_itemsets(self):
        return set(self.header_table.keys())
    
    
    def display_header_table(self):
        print(self.header_table.items())
    
    
    def get_ordered_itemsets(self,itemsets):
        sorted_items=sorted(itemsets.items(), key=operator.itemgetter(1), reverse=True)
        items=[item[0] for item in sorted_items]
        return items
    
    def update_fp_tree(self,tree_node,ordered_items,occurance_cnt):
        if ordered_items[0] in tree_node.child_node:
            tree_node.child_node[ordered_items[0]].update_item_frequency(occurance_cnt)
        else:
            
            tree_node.child_node[ordered_items[0]]=fp_tree_node(ordered_items[0],occurance_cnt,tree_node)
            
            if not self.header_table[ordered_items[0]][1]:
                self.header_table[ordered_items[0]][1]=tree_node.child_node[ordered_items[0]]
            else:
                #update headertable
                target_node=tree_node.child_node[ordered_items[0]]
                start_node=self.header_table[ordered_items[0]][1]
                
                while(start_node.next_node != None):
                    start_node=start_node.next_node
                start_node.next_node=target_node
        
        if(len(ordered_items)>1):
            self.update_fp_tree(tree_node.child_node[ordered_items[0]],ordered_items[1::],occurance_cnt)
    
    def remove_infrequent_items(self):
        for item in list(self.header_table.keys()):
            if(self.header_table[item] < self.min_support):
                del self.header_table[item]
   

    def create_tree(self):
        for transaction in self.transactions:
            for item in transaction:
                self.header_table[item]=self.header_table.get(item,0) + self.transactions[transaction]
        
        self.remove_infrequent_items()
        
        frequent_items=self.get_frequent_itemsets()
        if(len(frequent_items)==0):
            return None,None
        
        for keys in self.header_table:
            self.header_table[keys]= [self.header_table[keys],None]
        
        self.fp_root=fp_tree_node('Φ',1,None)
        
        for transaction,transaction_occurances in self.transactions.items():
            temp_transactions=defaultdict()
            
            for item in transaction:
                if item in frequent_items:
                    temp_transactions[item]=self.header_table[item][0]
            
            if len(temp_transactions)>0:
                ordered_items=self.get_ordered_itemsets(temp_transactions)
                self.update_fp_tree(self.fp_root,ordered_items,transaction_occurances)
        return self.fp_root,self.header_table
    

class MineFPTree:
    def __init__(self,min_support):
        self.min_support=min_support
        self.frequent_itemset_list=list()
        
    def get_ordered_itemsets(self,itemsets):
        sorted_items=sorted(itemsets,key=lambda key : itemsets[key][0])
        return sorted_items
        #return [v[0] for v in sorted(itemsets.items(), key=lambda p: p[1][0])]
    
    def find_root(self,first_node,prefix_path):
        if(first_node.parent is not None):
            
            prefix_path.append(first_node.item_name)
            
            self.find_root(first_node.parent,prefix_path)
    
    def find_prefix_path(self,item_name,node):

        conditional_patterns={}
        
        while(node is not None):
            prefix_path=list()
            self.find_root(node,prefix_path)
            
            if(len(prefix_path)>1):
                conditional_patterns[frozenset(prefix_path[1:])]=node.frequency
            node=node.next_node
        return conditional_patterns
    
    def find_frequent_itemsets(self,start_node,header_tab,prefix_set):
       
        sorted_items=self.get_ordered_itemsets(header_tab)
        for base in sorted_items:
            temp_frequent_set=prefix_set.copy()
            temp_frequent_set.add(base)
            
            self.frequent_itemset_list.append(temp_frequent_set)
            
            conditional_base=self.find_prefix_path(base,header_tab[base][1])
            
            fp_tree_obj=build_fp_tree(conditional_base,self.min_support)
            
            temp_tree,temp_header_table=fp_tree_obj.create_tree()
  
            if(temp_header_table is not None):
                self.find_frequent_itemsets(temp_tree,temp_header_table,temp_frequent_set)


def load_transaction(data_path):
    transactions=[]
    with open(data_path,'r') as data_file:
        for lines in tqdm(data_file):
            transactions_list=list(lines.strip().split())
            transactions_list=[int(x) for x in transactions_list]

            transactions_list=list(np.unique(transactions_list))
            transactions_list.sort()
            transactions.append(transactions_list)
    
    transactions_dict=defaultdict()
    for transaction in transactions:
        transactions_dict[frozenset(transaction)]=1
    return transactions_dict


data_path=r'retail_data.txt'


transaction_dict=load_transaction(data_path)

fp_tree_object=build_fp_tree(transaction_dict,4170)
fp_tree_root,header_table=fp_tree_object.create_tree()


mine_tree=MineFPTree(4170)
mine_tree.find_frequent_itemsets(fp_tree_root,header_table,set([]))
print(mine_tree.frequent_itemset_list)