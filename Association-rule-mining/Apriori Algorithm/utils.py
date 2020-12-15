import numpy as np
from tqdm import tqdm

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


def count_occurances(itemsets,transactions):
	count=0
	for i in range(len(transactions)):
		if set(itemsets).issubset(set(transactions[i])):
			count+=1
	return count


def join_two_itemsets(itemset1,itemset2,order):
	itemset1.sort(key=lambda val : self.order.index(val))
	itemset2.sort(key=lambda val : self.order.index(val))
	
	for i in range(len(itemset1)-1):
		if itemset1[i]!=itemset2[i]:
			return []

	if order.index(itemset1[-1]) < order.index(itemset2[-1]):
		return itemset1+[itemset2[-1]]

	return []

def join_set_itemsets(set_of_itemsets,order):
	Candidates=[]
	for i in range(len(set_of_itemsets)):
		for j in range(i+1,len(set_of_itemsets)):
			itemsets_out=join_two_itemsets(set_of_itemsets[i],set_of_itemsets[j],order)
			if len(itemsets_out)>0:
				Candidates.append(itemsets_out)
	return Candidates

#initialization
def get_frequent(itemsets,transactions,minimum_support,prev_discarded):
	L=[]
	supp_count=[]
	new_discarded=[]

	num_transactions=len(transactions)

	k=len(prev_discarded.keys())

	for s in tqdm(range(len(itemsets))):
		discarded_before=False

		if k > 0:
			for item in prev_discarded[k]:
				if set(item).issubset(set(itemsets[s])):
					discarded_before=True
					break

		if not discarded_before:
			count=count_occurances(itemsets[s],transactions)
			if count/num_transactions >=minimum_support:
				L.append(itemsets[s])
				supp_count.append(count)
			else:
				new_discarded.append(itemsets[s])

	return L,supp_count,new_discarded