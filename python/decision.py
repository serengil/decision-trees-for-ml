import pandas as pd
import math

#------------------------

#df = pd.read_csv("golf.txt")
df = pd.read_csv("car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"])

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

	for i in range(0, columns-1):
		column_name = df.columns[i]
		classes = df[column_name].value_counts()
		
		gain = entropy * 1
		
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
			
		gains.append(gain)

	winner_index = gains.index(max(gains))
	winner_name = df.columns[winner_index]

	return winner_name

def buildDecisionTree(df,precondition):
	winner_name = findDecision(df)
	#print("winner is ",winner_name)

	classes = df[winner_name].value_counts().keys().tolist()

	for i in range(0,len(classes)):
		current_class = classes[i]
		subdataset = df[df[winner_name] == current_class]
		subdataset = subdataset.drop(columns=[winner_name])
		
		#print(subdataset)
		
		if len(subdataset['Decision'].value_counts().tolist()) == 1:
			final_decision = subdataset['Decision'].value_counts().keys().tolist()[0]
			print(precondition,"if ",winner_name," is ",current_class," then decision is ",final_decision)
		else:
			precondition = precondition + "if "+winner_name+" is "+current_class+" AND "
			#print("if ",winner_name," is ",current_class," AND ")
			buildDecisionTree(subdataset,precondition)
			precondition = ''
			

buildDecisionTree(df,'')	