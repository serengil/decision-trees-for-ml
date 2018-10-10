import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

#--------------------------------
#parameters

is_regression = True
plotTree = False

#--------------------------------
#data set
#dataset = pd.read_csv('dataset/golf.txt')
dataset = pd.read_csv('dataset/golf2.txt')
#dataset = pd.read_csv('dataset/iris.data', names=["Sepal length","Sepal width","Petal length","Petal width","Decision"])

#--------------------------------
#label encoding
#you must convert both categorical features to int in LightGBM
print("label encoding procedures:")

features = []; categorical_features = []

num_of_rows = dataset.shape[0]
num_of_columns = dataset.shape[1]
num_of_classes = 1 #default for regression

for i in range(0, num_of_columns):
	column_name = dataset.columns[i]
	column_type = dataset[column_name].dtypes
	
	if i != num_of_columns - 1: #skip target
		features.append(column_name)
	
	if column_type == 'object':
		print(column_name,": ", end='')
		feature_classes = dataset[column_name].unique()
		#print(feature_classes)
		
		if is_regression == False and i == num_of_columns - 1:
			num_of_classes = len(feature_classes)
		
		for j in range(len(feature_classes)):
			feature_class = feature_classes[j]
			print(feature_class," -> ",j,", ",end='')
						
			dataset[column_name] = dataset[column_name].replace(feature_class, str(j))
		
		if i != num_of_columns - 1: #skip target
			categorical_features.append(column_name)
		
		print("")

print("num_of_classes: ",num_of_classes)
print("features: ",features)
print("categorical features: ",categorical_features)
print("\nencoded dataset:\n",dataset.head())			

#--------------------------------

target_name = dataset.columns[num_of_columns - 1]

y_train = dataset[target_name].values
x_train = dataset.drop(columns=[target_name]).values

#print(x_train); print(y_train)

lgb_train = lgb.Dataset(x_train, y_train
	,feature_name = features
	, categorical_feature = categorical_features
)

params = {
	'task': 'train'
	, 'boosting_type': 'gbdt'
	, 'objective': 'regression' if is_regression == True else 'multiclass'
	, 'num_class': num_of_classes
	, 'metric': 'rmsle' if is_regression == True else 'multi_logloss'
	, 'min_data': 1
	#, 'learning_rate':0.1
	, 'verbose': -1
}

gbm = lgb.train(params
	, lgb_train
	, num_boost_round=50
	#, valid_sets=lgb_eval
)

predictions = gbm.predict(x_train)

#print(predictions)
"""for i in predictions:
	print(np.argmax(i))"""

#--------------------------------

for index, instance in dataset.iterrows():
	actual = instance[target_name]
	
	if is_regression == True:
		prediction = round(predictions[index])
	else: #classification
		prediction = np.argmax(predictions[index])
	
	print((index+1),". actual= ",actual,", prediction= ",prediction)
	
if plotTree == True:
	ax = lgb.plot_tree(gbm)
	plt.show()
	
	#ax = lgb.plot_importance(gbm, max_num_features=10)
	#plt.show()