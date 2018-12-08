import pandas as pd
import math
import numpy as np
import time
import imp
#------------------------

algorithm = "C4.5" #ID3, C4.5, CART, Regression

#------------------------
#parameters

enableRandomForest = False
num_of_trees = 3 #this should be a prime number
enableMultitasking = False

dump_to_console = False #Set this True to print rules in console. Set this False to store rules in a flat file.

enableGradientBoosting = False
epochs = 10
learning_rate = 1

enableAdaboost = False
#------------------------
#Data set
df = pd.read_csv("dataset/golf.txt") #nominal features and target
#df = pd.read_csv("dataset/golf2.txt") #nominal and numeric features, nominal target
#df = pd.read_csv("dataset/golf3.txt") #nominal features and numeric target
#df = pd.read_csv("dataset/golf4.txt") #nominal and numeric features, numeric target
#df = pd.read_csv("dataset/car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"])
#df = pd.read_csv("dataset/iris.data", names=["Sepal length","Sepal width","Petal length","Petal width","Decision"])
#df = pd.read_csv("dataset/adaboost.txt")
#you can find these data sets at https://github.com/serengil/decision-trees-for-ml/tree/master/dataset

dataset = df.copy()
#------------------------

if algorithm == 'Regression':
	if df['Decision'].dtypes == 'object':
		raise ValueError('Regression trees cannot be applied for nominal target values! You can either change the algorithm or data set.')

if df['Decision'].dtypes != 'object': #this must be regression tree even if it is not mentioned in algorithm
	algorithm = 'Regression'
	global_stdev = df['Decision'].std(ddof=0)

if enableGradientBoosting == True:
	dump_to_console = False
	algorithm = 'Regression'
	"""if algorithm != 'Regression':
		raise ValueError('gradient boosting must be applied for regression problems (for now). Change the data set.')"""

print(algorithm," tree is going to be built...")

dataset_features = dict() #initialize a dictionary. this is going to be used to check features numeric or nominal. numeric features should be transformed to nominal values based on scales.
#------------------------

def softmax(w):
	e = np.exp(np.array(w))
	dist = e / np.sum(e)
	return dist

def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0
	
def processContinuousFeatures(df, column_name, entropy):
	unique_values = sorted(df[column_name].unique())
	#print(column_name,"->",unique_values)
	
	subset_gainratios = []; subset_gains = []; subset_ginis = []; subset_red_stdevs = []
	
	for i in range(0, len(unique_values)-1):
		threshold = unique_values[i]
		
		subset1 = df[df[column_name] <= threshold]
		subset2 = df[df[column_name] > threshold]
		
		subset1_rows = subset1.shape[0]; subset2_rows = subset2.shape[0]
		total_instances = df.shape[0] #subset1_rows+subset2_rows
		
		subset1_probability = subset1_rows / total_instances
		subset2_probability = subset2_rows / total_instances
		
		if algorithm == 'ID3' or algorithm == 'C4.5':
			threshold_gain = entropy - subset1_probability*calculateEntropy(subset1) - subset2_probability*calculateEntropy(subset2)
			subset_gains.append(threshold_gain)
		
		if algorithm == 'C4.5': #C4.5 also need gain in the block above. That's why, instead of else if we used direct if condition here
			threshold_splitinfo = -subset1_probability * math.log(subset1_probability, 2)-subset2_probability*math.log(subset2_probability, 2)
			gainratio = threshold_gain / threshold_splitinfo
			subset_gainratios.append(gainratio)
				
		elif algorithm == 'CART':
			decision_for_subset1 = subset1['Decision'].value_counts().tolist()
			decision_for_subset2 = subset2['Decision'].value_counts().tolist()
			
			gini_subset1 = 1; gini_subset2 = 1
			
			for j in range(0, len(decision_for_subset1)):
				gini_subset1 = gini_subset1 - math.pow((decision_for_subset1[j]/subset1_rows),2)
			
			for j in range(0, len(decision_for_subset2)):
				gini_subset2 = gini_subset2 - math.pow((decision_for_subset2[j]/subset2_rows),2)
			
			gini = (subset1_rows/total_instances)*gini_subset1 + (subset2_rows/total_instances) * gini_subset2
			
			subset_ginis.append(gini)
		
		#----------------------------------
		elif algorithm == 'Regression':
			superset_stdev = df['Decision'].std(ddof=0)
			subset1_stdev = subset1['Decision'].std(ddof=0)
			subset2_stdev = subset2['Decision'].std(ddof=0)
			
			threshold_weighted_stdev = (subset1_rows/total_instances)*subset1_stdev + (subset2_rows/total_instances)*subset2_stdev
			threshold_reducted_stdev = superset_stdev - threshold_weighted_stdev
			subset_red_stdevs.append(threshold_reducted_stdev)
			
		#----------------------------------
	
	if algorithm == "C4.5":
		winner_one = subset_gainratios.index(max(subset_gainratios))
	elif algorithm == "ID3": #actually, ID3 does not support for continuous features but we can still do it
		winner_one = subset_gains.index(max(subset_gains))
	elif algorithm == "CART":
		winner_one = subset_ginis.index(min(subset_ginis))
	elif algorithm == "Regression":
		winner_one = subset_red_stdevs.index(max(subset_red_stdevs))
		
	winner_threshold = unique_values[winner_one]
	
	#print("theshold is ",winner_threshold," for ",column_name)
	df[column_name] = np.where(df[column_name] <= winner_threshold, "<="+str(winner_threshold), ">"+str(winner_threshold))
	
	return df

def calculateEntropy(df):
	
	if algorithm == 'Regression':
		return 0
	
	#print(df)

	instances = df.shape[0]; columns = df.shape[1]
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

	columns = df.shape[1]; instances = df.shape[0]

	gains = []; gainratios = []; ginis = []; reducted_stdevs = []

	for i in range(0, columns-1):
		column_name = df.columns[i]
		column_type = df[column_name].dtypes
		
		#print(column_name,"->",column_type)
		
		if column_type != 'object':
			df = processContinuousFeatures(df, column_name, entropy)
		
		classes = df[column_name].value_counts()
		
		gain = entropy * 1; splitinfo = 0; gini = 0; weighted_stdev = 0
		
		for j in range(0, len(classes)):
			current_class = classes.keys().tolist()[j]
			#print(column_name,"->",current_class)
			
			subdataset = df[df[column_name] == current_class]
			#print(subdataset)
			
			subset_instances = subdataset.shape[0]
			class_probability = subset_instances/instances
			
			if algorithm == 'ID3' or algorithm == 'C4.5':
				subset_entropy = calculateEntropy(subdataset)
				#print("entropy for this sub dataset is ", subset_entropy)
				gain = gain - class_probability * subset_entropy			
			
			if algorithm == 'C4.5':
				splitinfo = splitinfo - class_probability*math.log(class_probability, 2)
			
			elif algorithm == 'CART': #GINI index
				decision_list = subdataset['Decision'].value_counts().tolist()
				
				subgini = 1
				
				for k in range(0, len(decision_list)):
					subgini = subgini - math.pow((decision_list[k]/subset_instances), 2)
				
				gini = gini + (subset_instances / instances) * subgini
			
			elif algorithm == 'Regression':
				subset_stdev = subdataset['Decision'].std(ddof=0)
				weighted_stdev = weighted_stdev + (subset_instances/instances)*subset_stdev
		
		#iterating over classes for loop end
		#-------------------------------
		
		if algorithm == "ID3":
			gains.append(gain)
		
		elif algorithm == "C4.5":
		
			if splitinfo == 0:
				splitinfo = 100 #this can be if data set consists of 2 rows and current column consists of 1 class. still decision can be made (decisions for these 2 rows same). set splitinfo to very large value to make gain ratio very small. in this way, we won't find this column as the most dominant one.
				
			gainratio = gain / splitinfo
			gainratios.append(gainratio)
		
		elif algorithm == "CART":
			ginis.append(gini)
		
		elif algorithm == 'Regression':
			reducted_stdev = stdev - weighted_stdev
			reducted_stdevs.append(reducted_stdev)
	
	#print(df)
	if algorithm == "ID3":
		winner_index = gains.index(max(gains))
	elif algorithm == "C4.5":
		winner_index = gainratios.index(max(gainratios))
	elif algorithm == "CART":
		winner_index = ginis.index(min(ginis))
	elif algorithm == "Regression":
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
	
	#find winner index, this cannot be returned by find decision because columns dropped in previous steps
	j = 0 
	for i in dataset_features:
		if i == winner_name:
			winner_index = j
		j = j + 1
	
	numericColumn = False
	if dataset_features[winner_name] != 'object':
		numericColumn = True
	
	#restoration
	columns = df.shape[1]
	for i in range(0, columns-1):
		column_name = df.columns[i]; column_type = df[column_name].dtypes
		if column_type != 'object' and column_name != winner_name:
			df[column_name] = df_copy[column_name]
	
	classes = df[winner_name].value_counts().keys().tolist()

	for i in range(0,len(classes)):
		current_class = classes[i]
		subdataset = df[df[winner_name] == current_class]
		subdataset = subdataset.drop(columns=[winner_name])
		
		if numericColumn == True:
			compareTo = current_class #current class might be <=x or >x in this case
		else:
			compareTo = " == '"+str(current_class)+"'"
		
		#print(subdataset)
		
		terminateBuilding = False
		
		#-----------------------------------------------
		#can decision be made?
		
		if enableAdaboost == True:
			#final_decision = subdataset['Decision'].value_counts().idxmax()
			final_decision = subdataset['Decision'].mean() #get average
			terminateBuilding = True
		elif len(subdataset['Decision'].value_counts().tolist()) == 1:
			final_decision = subdataset['Decision'].value_counts().keys().tolist()[0] #all items are equal in this case
			terminateBuilding = True
		elif subdataset.shape[1] == 1: #if decision cannot be made even though all columns dropped
			final_decision = subdataset['Decision'].value_counts().idxmax() #get the most frequent one
			terminateBuilding = True
		elif algorithm == 'Regression' and subdataset.shape[0] < 5: #pruning condition
		#elif algorithm == 'Regression' and subdataset['Decision'].std(ddof=0)/global_stdev < 0.4: #pruning condition
			final_decision = subdataset['Decision'].mean() #get average
			terminateBuilding = True
		#-----------------------------------------------
		
		if dump_to_console == True:
			print(formatRule(root),"if ",winner_name,compareTo,":")
		else:
			#storeRule(file,(formatRule(root),"if ",winner_name,compareTo,":"))
			storeRule(file,(formatRule(root),"if obj[",str(winner_index),"]",compareTo,":"))
		
		#-----------------------------------------------
		
		if terminateBuilding == True: #check decision is made
			if dump_to_console == True:
				print(formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp)
			else:
				storeRule(file,(formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp))
		else: #decision is not made, continue to create branch and leafs
			root = root + 1 #the following rule will be included by this rule. increase root
			buildDecisionTree(subdataset,root,file)
		
		root = tmp_root * 1
		
#--------------------------

if(True): #header of rules files
	header = "def findDecision("
	num_of_columns = df.shape[1]-1
	for i in range(0, num_of_columns):
		if dump_to_console == True:
			if i > 0:
				header = header + ","
			header = header + df.columns[i]
		
		column_name = df.columns[i]
		dataset_features[column_name] = df[column_name].dtypes

	if dump_to_console == False:
		header = header + "obj"
		
	header = header + "):\n"

	if dump_to_console == True:
		print(header,end='')

#--------------------------
begin = time.time()

if enableAdaboost == True:
	
	rows = df.shape[0]; columns = df.shape[1]
	final_predictions = pd.DataFrame(np.zeros([rows, 1]), columns=['prediction'])
	
	worksheet = df.copy()
	worksheet['weight'] = 1 / rows
	
	tmp_df = df.copy()
	tmp_df['Decision'] = worksheet['weight'] * tmp_df['Decision'] #normal distribution
	
	for i in range(0, 4):	
		root = 1
		file = "rules_"+str(i)+".py"
		
		if dump_to_console == False: createFile(file, header)
		
		print(tmp_df)
		buildDecisionTree(tmp_df,root,file)
		
		moduleName = "rules_"+str(i)
		fp, pathname, description = imp.find_module(moduleName)
		rules = imp.load_module(moduleName, fp, pathname, description) #rules0
		
		predictions = []; losses = []
		
		for row, instance in dataset.iterrows():
			features = []
			for j in range(0, columns):
				features.append(instance[j])
			
			prediction = rules.findDecision(features)
			actual = instance['Decision']
			#print(actual," - ",prediction)
			
			prediction = sign(prediction)
			actual = sign(actual)
			#print(actual," - ",prediction)
			print(prediction)
			
			if actual == prediction: loss = 0
			else: loss = 1
			
			predictions.append(prediction)
			losses.append(loss)
			
		worksheet['prediction'] = pd.Series(predictions).values
		worksheet['loss'] = pd.Series(losses).values
		
		worksheet['w*l'] = worksheet['weight'] * worksheet['loss']
		error = worksheet['w*l'].sum()
		alpha = math.log((1-error)/error)/2
		worksheet['alpha'] = alpha
		
		final_predictions['prediction'] = final_predictions['prediction'] + alpha * worksheet['prediction']
		
		print("error in this round: ",error)
		print("alpha in this round: ",alpha)
		
		#worksheet['weight']*math.exp(-worksheet['alpha']*worksheet['Decision']*worksheet['prediction'])
		
		worksheet['weight_t+1'] = worksheet['weight']*np.exp(-worksheet['alpha']*worksheet['Decision']*worksheet['prediction'])
		
		#normalize
		worksheet['weight_t+1'] = worksheet['weight_t+1'] / worksheet['weight_t+1'].sum()
		
		print(worksheet)
		
		tmp_df = df.copy()
		tmp_df['Decision'] = worksheet['weight_t+1'] * tmp_df['Decision']
		worksheet['weight'] = worksheet['weight_t+1']
		
		print("-------------------------")
	
	print(final_predictions)
	
	for row, instance in final_predictions.iterrows():
		print("actual: ",df.loc[row]['Decision'],", prediction: ",sign(instance['prediction'])," (",df.loc[row]['Decision'] == sign(instance['prediction']),")")
	
elif enableGradientBoosting == True:
	
	if df['Decision'].dtypes == 'object': #transform classification problem to regression
		
		print("gradient boosting for classification")
		temp_df = df.copy()
		original_dataset = df.copy()
		worksheet = df.copy()
		
		classes = df['Decision'].unique()
		
		boosted_predictions = np.zeros([df.shape[0], len(classes)])
		
		for epoch in range(0, epochs):
			for i in range(0, len(classes)):
				current_class = classes[i]
				
				if epoch == 0:
					temp_df['Decision'] = np.where(df['Decision'] == current_class, 1, 0)
					worksheet['Y_'+str(i)] = temp_df['Decision']
				else:
					temp_df['Decision'] = worksheet['Y-P_'+str(i)]
				
				predictions = []
				
				#change data type for decision column
				temp_df[['Decision']].astype('int64')
				
				root = 1
				file = "rules-for-"+current_class+".py"
				
				if dump_to_console == False: createFile(file, header)
				
				buildDecisionTree(temp_df,root,file)
				#decision rules created
				#----------------------------
				
				#dynamic import
				moduleName = "rules-for-"+current_class
				fp, pathname, description = imp.find_module(moduleName)
				myrules = imp.load_module(moduleName, fp, pathname, description) #rules0
				
				num_of_columns = df.shape[1]
				
				for row, instance in df.iterrows():
					features = []
					for j in range(0, num_of_columns-1): #iterate on features
						features.append(instance[j])
					
					actual = temp_df.loc[row]['Decision']
					prediction = myrules.findDecision(features)
					predictions.append(prediction)
						
				#----------------------------
				if epoch == 0:
					worksheet['F_'+str(i)] = 0
				else:
					worksheet['F_'+str(i)] = pd.Series(predictions).values
					
				boosted_predictions[:,i] = boosted_predictions[:,i] + worksheet['F_'+str(i)].values
				
				worksheet['P_'+str(i)] = 0
				
				#----------------------------
				temp_df = df.copy() #restoration
			
			for row, instance in worksheet.iterrows():
				f_scores = []
				for i in range(0, len(classes)):
					f_scores.append(instance['F_'+str(i)])
				
				probabilities = softmax(f_scores)
				
				for j in range(0, len(probabilities)):
					instance['P_'+str(j)] = probabilities[j]
				
				worksheet.loc[row] = instance
			
			for i in range(0, len(classes)):
				worksheet['Y-P_'+str(i)] = worksheet['Y_'+str(i)] - worksheet['P_'+str(i)]
			
			print("round ",epoch+1)
			"""print(worksheet.head())
			print("------------------")"""
			
		"""print("boosted predictions:")
		for i in range(0, boosted_predictions.shape[0]):
			max_index = np.argmax(boosted_predictions[i])
			print(max_index)"""
			
	else: #regression problem
		root = 1
		file = "rules0.py"
		if dump_to_console == False:
			createFile(file, header)
		
		buildDecisionTree(df,root,file) #generate rules0
		
		#------------------------------
		
		for index in range(1,epochs):	
			#run data(i-1) and rules(i-1), save data1
			
			#dynamic import
			moduleName = "rules%s" % (index-1)
			fp, pathname, description = imp.find_module(moduleName)
			myrules = imp.load_module(moduleName, fp, pathname, description) #rules0
			
			new_data_set = "data%s.csv" % (index)
			f = open(new_data_set, "w")
			
			#put header in the following file
			columns = df.shape[1]
			
			for i, instance in df.iterrows():
				params = []
				line = ""
				for j in range(0, columns-1):
					params.append(instance[j])
					if j > 0:
						line = line + ","
					line = line + instance[j]
				
				
				prediction = myrules.findDecision(params) #apply rules(i-1) for data(i-1)
				actual = instance[columns-1]
				
				print(prediction)
				
				#loss was ((actual - prediction)^2) / 2
				#partial derivative of loss function with respect to the prediction is prediction - actual
				#y' = y' - alpha * gradient = y' - alpha * (prediction - actual) = y' = y' + alpha * (actual - prediction)
				#whereas y' is prediction and alpha is learning rate
				gradient = learning_rate*(actual - prediction)
				
				instance[columns-1] = gradient
				
				df.loc[i] = instance
			
			df.to_csv(new_data_set, index=False)
			#data(i) created
			#---------------------------------
			
			file = "rules"+str(index)+".py"
			
			if dump_to_console == False:
				createFile(file, header)
			
			buildDecisionTree(df,root,file)
			#rules(i) created
			#---------------------------------
	
elif enableRandomForest == False: #standard decision tree building
	root = 1
	
	file = "rules.py"
	
	if dump_to_console == False:
		createFile(file, header)
	
	buildDecisionTree(df,root,file)
	
	print("finished in ",time.time() - begin," seconds")
	
else: #Random Forest
	
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
		
		if __name__ == '__main__': #windows returns expection if this control is not applied for multitasking
			with Pool(num_of_trees) as pool:
				pool.starmap(buildDecisionTree, subsets)
			
			print("finished in ",time.time() - begin," seconds")

