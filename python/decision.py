import pandas as pd
import math
import numpy as np
import time
#------------------------

algorithm = "C4.5" #ID3, C4.5, CART, Regression

enableRandomForest = False
num_of_trees = 5 #this should be a prime number
enableMultitasking = True

dump_to_console = True #Set this True to print rules in console. Set this False to store rules in a flat file.

#------------------------

#df = pd.read_csv("golf.txt")
df = pd.read_csv("golf2.txt")
#df = pd.read_csv("golf3.txt")
#df = pd.read_csv("car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"])
#df = pd.read_csv("iris.data", names=["Sepal length","Sepal width","Petal length","Petal width","Decision"])

#------------------------

if df['Decision'].dtypes != 'object':
	algorithm = 'Regression'
	global_stdev = df['Decision'].std(ddof=0)

dataset_features = dict()
#------------------------

def processContinuousFeatures(df, column_name, entropy):
	unique_values = sorted(df[column_name].unique())
	#print(column_name,"->",unique_values)
	
	subset_gainratios = []
	subset_gains = []
	subset_ginis = []
	
	for i in range(0, len(unique_values)-1):
		threshold = unique_values[i]
		
		subset1 = df[df[column_name] <= threshold]
		subset2 = df[df[column_name] > threshold]
		
		subset1_rows = subset1.shape[0]
		subset2_rows = subset2.shape[0]
		
		total_instances = df.shape[0] #subset1_rows+subset2_rows
		
		subset1_probability = subset1_rows / total_instances
		subset2_probability = subset2_rows / total_instances
		
		threshold_gain = entropy - subset1_probability*calculateEntropy(subset1) - subset2_probability*calculateEntropy(subset2)
		threshold_splitinfo = -subset1_probability * math.log(subset1_probability, 2)-subset2_probability*math.log(subset2_probability, 2)
		
		gainratio = threshold_gain / threshold_splitinfo
		subset_gainratios.append(gainratio)
		subset_gains.append(threshold_gain)
		
		#---------------------------------
		
		decision_for_subset1 = subset1['Decision'].value_counts().tolist()
		decision_for_subset2 = subset2['Decision'].value_counts().tolist()
		
		gini_subset1 = 1; gini_subset2 = 1
		
		for j in range(0, len(decision_for_subset1)):
			gini_subset1 = gini_subset1 - math.pow((decision_for_subset1[j]/subset1_rows),2)
		
		for j in range(0, len(decision_for_subset2)):
			gini_subset2 = gini_subset2 - math.pow((decision_for_subset2[j]/subset2_rows),2)
		
		gini = (subset1_rows/total_instances)*gini_subset1 + (subset2_rows/total_instances) * gini_subset2
		
		subset_ginis.append(gini)
	
	if algorithm == "C4.5":
		winner_one = subset_gainratios.index(max(subset_gainratios))
	elif algorithm == "ID3": #actually, ID3 does not support for continuous features but we can do it
		winner_one = subset_gains.index(max(subset_gains))
	elif algorithm == "CART":
		winner_one = subset_ginis.index(min(subset_ginis))
		
	winner_threshold = unique_values[winner_one]
	
	#print("theshold is ",winner_threshold," for ",column_name)
	df[column_name] = np.where(df[column_name] <= winner_threshold, "<="+str(winner_threshold), ">"+str(winner_threshold))
	
	return df

def calculateEntropy(df):
	
	if algorithm == 'Regression':
		return 0
	
	#print(df)

	instances = df.shape[0]
	columns = df.shape[1]
	#print(instances," rows, ",columns," columns")

	decisions = df['Decision'].value_counts().keys().tolist()

	entropy = 0

	for i in range(0, len(decisions)):
		decision = decisions[i]
		num_of_decisions = df['Decision'].value_counts().tolist()[i]
		#print(decision,"->",num_of_decisions)
		
		class_probability = num_of_decisions/instances
		
		entropy = entropy - class_probability*math.log(class_probability, 2)
		
	return entropy

def findDecision(df):
	if algorithm == 'Regression':
		stdev = df['Decision'].std(ddof=0)
		
	entropy = calculateEntropy(df)
	#print("entropy: ",entropy)

	columns = df.shape[1]
	instances = df.shape[0]

	gains = []
	gainratios = []
	ginis = []
	reducted_stdevs = []

	for i in range(0, columns-1):
		column_name = df.columns[i]
		column_type = df[column_name].dtypes
		
		#print(column_name,"->",column_type)
		
		if column_type != 'object':
			df = processContinuousFeatures(df, column_name, entropy)
		
		classes = df[column_name].value_counts()
		
		gain = entropy * 1
		splitinfo = 0
		gini = 0
		weighted_stdev = 0
		
		for j in range(0, len(classes)):
			current_class = classes.keys().tolist()[j]
			#print(column_name,"->",current_class)
			
			subdataset = df[df[column_name] == current_class]
			#print(subdataset)
			subset_entropy = calculateEntropy(subdataset)
			#print("entropy for this sub dataset is ", subset_entropy)
			
			subset_instances = subdataset.shape[0]
			class_probability = subset_instances/instances
			
			gain = gain - class_probability * subset_entropy			
			splitinfo = splitinfo - class_probability*math.log(class_probability, 2)
			
			#------------------------------
			#GINI index
			
			decision_list = subdataset['Decision'].value_counts().tolist()
			
			subgini = 1
			
			for k in range(0, len(decision_list)):
				subgini = subgini - math.pow((decision_list[k]/subset_instances), 2)
			
			gini = gini + (subset_instances / instances) * subgini
			
			#------------------------------
			
			if algorithm == 'Regression':
				subset_stdev = subdataset['Decision'].std(ddof=0)
				weighted_stdev = weighted_stdev + (subset_instances/instances)*subset_stdev
			
		if algorithm == "ID3":
			gains.append(gain)
		
		if algorithm == "C4.5":
		
			if splitinfo == 0:
				splitinfo = 100 #this can be if data set consists of 2 rows and current column consists of 1 class. still decision can be made (decisions for these 2 rows same). set splitinfo to very large value to make gain ratio very small. in this way, we won't find this column as the most dominant one.
				
			gainratio = gain / splitinfo
			gainratios.append(gainratio)
		
		if algorithm == "CART":
			ginis.append(gini)
		
		if algorithm == 'Regression':
			reducted_stdev = stdev - weighted_stdev
			reducted_stdevs.append(reducted_stdev)
	
	#print(df)
	if algorithm == "ID3":
		winner_index = gains.index(max(gains))
	elif algorithm == "C4.5":
		winner_index = gainratios.index(max(gainratios))
	elif algorithm == "CART":
		winner_index = ginis.index(min(ginis))
	elif algorithm == 'Regression':
		winner_index = reducted_stdevs.index(max(reducted_stdevs))
	winner_name = df.columns[winner_index]

	return winner_name
	
def formatRule(root):
	resp = ''
	
	for i in range(0, root):
		resp = resp + '   '
	
	return resp	

def storeRule(file,content):
		f = open(file, "a+")
		f.writelines(content)
		f.writelines("\n")

def createFile(file,content):
		f = open(file, "w")
		f.write(content)
	
def buildDecisionTree(df,root,file):
	#print(df.shape)
	charForResp = "'"
	if algorithm == 'Regression':
		charForResp = ""

	tmp_root = root * 1
	
	df_copy = df.copy()
	
	winner_name = findDecision(df)
	#print("winner is ",winner_name)
	
	numericColumn = False
	if dataset_features[winner_name] != 'object':
		numericColumn = True
	
	#restoration
	columns = df.shape[1]
	for i in range(0, columns-1):
		column_name = df.columns[i]
		if column_name != winner_name:
			df[column_name] = df_copy[column_name]
	
	classes = df[winner_name].value_counts().keys().tolist()

	for i in range(0,len(classes)):
		current_class = classes[i]
		subdataset = df[df[winner_name] == current_class]
		subdataset = subdataset.drop(columns=[winner_name])
		
		if numericColumn == True:
			compareTo = current_class
		else:
			compareTo = " == '"+str(current_class)+"'"
		
		#print(subdataset)
		
		if len(subdataset['Decision'].value_counts().tolist()) == 1:
			final_decision = subdataset['Decision'].value_counts().keys().tolist()[0]
			if dump_to_console == True:
				print(formatRule(root),"if ",winner_name,compareTo,":")
				print(formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp)
			else:
				storeRule(file,(formatRule(root),"if ",winner_name,compareTo,":"))
				storeRule(file,(formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp))
		elif subdataset.shape[1] == 1:
			final_decision = subdataset['Decision'].value_counts().idxmax()
			if dump_to_console == True:			
				print(formatRule(root),"if ",winner_name,compareTo,":")
				print(formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp)
			else:
				storeRule(file,(formatRule(root),"if ",winner_name,compareTo,":"))
				storeRule(file,(formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp))
		elif algorithm == 'Regression' and subdataset.shape[0] < 5:
		#elif algorithm == 'Regression' and subdataset['Decision'].std(ddof=0)/global_stdev < 0.4:
			final_decision = subdataset['Decision'].mean()
			if dump_to_console == True:
				print(formatRule(root),"if ",winner_name,compareTo,":")
				print(formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp)
			else:
				storeRule(file,(formatRule(root),"if ",winner_name,compareTo,":"))
				storeRule(file,(formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp))
		else:
			if dump_to_console == True:
				print(formatRule(root),"if ",winner_name,compareTo,":")
			else:
				storeRule(file,(formatRule(root),"if ",winner_name,compareTo,":"))
			root = root + 1
			buildDecisionTree(subdataset,root,file)
		
		root = tmp_root * 1
		
#--------------------------
header = "def findDecision("
num_of_columns = df.shape[1]-1
for i in range(0, num_of_columns):
	if i > 0:
		header = header + ","
	header = header + df.columns[i]
	
	column_name = df.columns[i]
	dataset_features[column_name] = df[column_name].dtypes

header = header + "):\n"

if dump_to_console == True:
	print(header,end='')

#--------------------------
begin = time.time()

if enableRandomForest == False:
	root = 1
	
	file = "rules.py"
	
	if dump_to_console == False:
		createFile(file, header)
	
	buildDecisionTree(df,root,file)
	print("finished in ",time.time() - begin," seconds")
	
else: 
	
	if enableMultitasking == False: #serial
		
		for i in range(0, num_of_trees):
			subset = df.sample(frac=1/num_of_trees)
			
			root = 1
			
			file = "rule_"+str(i)+".py"
			
			if dump_to_console == False:
				createFile(file, header)
			
			buildDecisionTree(subset,root, file)
		
		print("finished in ",time.time() - begin," seconds")
		
	else: #parallel
		
		from multiprocessing import Pool
		
		subsets = []
		
		for i in range(0, num_of_trees):
			
			file = "rule_"+str(i)+".py"
			
			subset = df.sample(frac=1/num_of_trees)
			root = 1
			subsets.append((subset, root, file))
			
			if dump_to_console == False:
				createFile(file, header)
		
		if __name__ == '__main__':
			with Pool(num_of_trees) as pool:
				pool.starmap(buildDecisionTree, subsets)
			
			print("finished in ",time.time() - begin," seconds")
		
