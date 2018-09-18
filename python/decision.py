import pandas as pd
import math
import numpy as np
#------------------------

algorithm = "C4.5" #ID3

#df = pd.read_csv("golf.txt")
df = pd.read_csv("golf2.txt")
#df = pd.read_csv("car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"])

def processContinuousFeatures(df, column_name, entropy):
	unique_values = sorted(df[column_name].unique())
	#print(column_name,"->",unique_values)
	
	subset_gainratios = []
	
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
	
	max_one = subset_gainratios.index(max(subset_gainratios))
	winner_threshold = unique_values[max_one]
	
	#print("theshold is ",winner_threshold," for ",column_name)
	df[column_name] = np.where(df[column_name] <= winner_threshold, "<="+str(winner_threshold), ">"+str(winner_threshold))
	
	return df

def calculateEntropy(df):
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
	entropy = calculateEntropy(df)
	#print("entropy: ",entropy)

	columns = df.shape[1]
	instances = df.shape[0]

	gains = []
	gainratios = []

	for i in range(0, columns-1):
		column_name = df.columns[i]
		column_type = df[column_name].dtypes
		
		#print(column_name,"->",column_type)
		
		if column_type != 'object':
			df = processContinuousFeatures(df, column_name, entropy)
		
		classes = df[column_name].value_counts()
		
		gain = entropy * 1
		splitinfo = 0
		
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
			
		gains.append(gain)
		
		gainratio = gain / splitinfo
		gainratios.append(gainratio)
	
	print(df)
	if algorithm == "ID3":
		winner_index = gains.index(max(gains))
	elif algorithm == "C4.5":
		winner_index = gainratios.index(max(gainratios))
	winner_name = df.columns[winner_index]

	return winner_name

def buildDecisionTree(df,precondition):
	
	df_copy = df.copy()
	
	winner_name = findDecision(df)
	#print("winner is ",winner_name)
	
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
		
		#print(subdataset)
		
		if len(subdataset['Decision'].value_counts().tolist()) == 1:
			final_decision = subdataset['Decision'].value_counts().keys().tolist()[0]
			print(precondition,"if ",winner_name," is ",str(current_class)," then decision is ",final_decision)
		else:
			precondition = precondition + "if "+winner_name+" is "+str(current_class)+" AND "
			#print("if ",winner_name," is ",current_class," AND ")
			buildDecisionTree(subdataset,precondition)
			precondition = ''
			

buildDecisionTree(df,'')	
