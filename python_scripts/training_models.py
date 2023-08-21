import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Sequence
import typing
from ruamel import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.stats import ks_2samp  #compute KS distance

import csv
import gc
import datetime
import glob
import os
import random

#1. run R script to generate data


def load_experiment(path_to_experiment):
	#load experiment parameters from the yaml file
	data = yaml.safe_load(open(path_to_experiment))
	return data


class PreprocessData:
	def __init__(self, path):
		self.data_path = path #path for the data file
		self.label = 'Dx_OpioidOverdose_0to1_Y' #outcome label: opioid overdose in the past one year


	def get_cohorts(self):
		'''
		This function generates a case-control cohort for each experiment. 
		The cohort is drawn from a dataset of 20,000,000 observations.

		First, all positive observations are selected based on the 'Dx_OpioidOverdose_0to1_Y' column.
		Then, negative observations are randomly selected to form the cohort.
		The ratio of positive to negative observations in each cohort is 1:100.
		The output is a dataframe containing the combined positive and negative observations.

		'''

		#Step 1: load dataset
		data = pd.read_csv(self.data_path + "simulated_data_big_sample.csv")
		# change input to shifted dataset 
		#data = pd.read_csv(self.data_path + "simulated_data_big_sample_shuffled_outcome.csv")
        

		#Step 2: Select the positive cases
		positive = data.loc[data['Dx_OpioidOverdose_0to1_Y'] == 1]

		#Select the negative cases and draw 100 observations for the cohort
		negative = data.loc[data['Dx_OpioidOverdose_0to1_Y'] == 0]
		negative_cohort = negative.sample(n=positive.shape[0]*100, random_state=random.randint(0,10000))

		#Step 3: Combine the negative observations with the positive observations to form the case-control cohort.
		case_control_cohort = pd.concat([positive,negative_cohort])

		return case_control_cohort


	def pre_process(self): 
		"""
		This function generates a feature matrix and a vector of outcome labels for model training.

		First, a cohort is drawn from the entire dataset.
		Then, the feature matrix is separated from the label vector.

		"""

		#Draw a cohort from the entire dataset using the get_cohort function.
		data = self.get_cohorts()
        
        #Seperate the feature matrix.
		X = data.drop(columns=[self.label])

		#Seperate the outcome label.
		y = data[self.label]

		return X, y

	def get_train_test_split(self):
		''' 
		This function splits the feature matrix and outcome variable, into train and test sets, with a ratio of 0.75:0.25 using stratified splitting.

		Returns:
		X_train: Feature matrix of the training set
		X_test: Feature matrix of the test set
		y_train: Outcome variable of the training set
		y_test: Outcome variable of the test set
		'''


		#Get the feature matrix and outcome variable from the pre_process function.
		X, y= self.pre_process()

		#  Split the feature matrix and outcome variable into train and test sets, with a test size of 0.25.
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y)
		
		return X_train, X_test, y_train, y_test



class ColumnSelector:
	'''
	ColumnSelector class is a feature selector for the pipeline, which works with pandas DataFrame format. 
	Here we select the features used in the pipeline. The feature groups are listed in the experiment.yaml file
	 '''
	
	def __init__(self, columns):
		#Stores the columns to be selected for training
		self.columns = columns 

	def fit(self, X, y=None):
		# fit the data into columnSelector class
		return self

	def transform(self, X):
		'''Transforms the input DataFrame (a feature matrix), by selecting the desired columns.'''

		#  AssertIsInstance() in Python is a unittest library function to check whether an object is an instance of a given class or not. Here we check whether the object is a Pandas dataFrame
		assert isinstance(X, pd.DataFrame)

		#Check if feature matrix contains specified column
		try:
			return X[self.columns]
		# Raises an error if the DataFrame does not include the columns
		except KeyError:
			cols_error = list(set(self.columns) - set(X.columns))
			raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

	# def get_feature_names(self):
	# 	## Returns the list of selected column names
	# 	return self.columns.tolis


class Training:
	"""
	The Training class contains a training pipeline, which includes processing feature sets selected in experiment.yaml. The feature processing includes normalization and imputation of the features. The model was trained with k fold cross validation. Finally, the loss between groups defined by sensitive attributes is compared.
	"""

	def __init__(self, X_train, X_test, y_train, y_test, parameters, features_list):

		'''
		The __init__ function initializes the variables passed into the functions within the class. 
		- X_train: Train feature
		- y_train: Train outcome
		- X_test: Test feature
		- y_test: Test outcome
		- parameters: Parameters, as specified in experiment.yaml
		- label: Outcome variable name ('Dx_OpioidOverdose_0to1_Y')
		- features_list: List of features included in training, as specified in experiment.yaml
		'''

		self.X_train = X_train 
		self.y_train = y_train 
		self.X_test = X_test 
		self.y_test = y_test 
		self.parameters = parameters
		#self.path = "/Users/luciachen/Dropbox/simulated_data/" #path for the simulated dataset
		self.label = 'Dx_OpioidOverdose_0to1_Y'
		self.features_list = features_list


	def get_feature_names(self):
		"""
		This function retrieves feature column names from the feature matrix based on the feature_list defined in the experiment.yaml file.
		"""

		fea_list = []
		#Iterate over each feature in the feature list
		for fea in self.features_list: 
			# Retrieve the column names that contain the current feature in the train feature matrix
			f_list = [i for i in self.X_train.columns if fea in i]
			#Append the column names to the feature list
			fea_list.append(f_list)

		#Flatten the feature list
		flat = [x for sublist in fea_list for x in sublist]
        
		#Returns flattened list ['feature1', 'feature2', ...]
		return flat



	def setup_pipeline(self, classifier):
		'''
		This function sets up the pipeline for feature selection and classification
		'''

		#Select the feature names from the experiment.yaml file
		features_col = self.get_feature_names()

		#Pipeline for feature selection
		pipeline = Pipeline([

		('other_features', Pipeline([
				
				#Insert the selected feature sets into the pipeline
				('selector', ColumnSelector(columns=features_col)),

				## Impute missing values with mean
				('impute', SimpleImputer(strategy='mean')),
			 ])),
        
		#Pipeline for scaling features and classification
		('clf', Pipeline([
			    
			   ('scale', StandardScaler(with_mean=False)),  # Scale the features
				('classifier', classifier),  # Use the specified classifier defined in experiment.yaml
		   
				 ])),
		])
		return pipeline

	def training_models(self, pipeline):

		'''This function is used to train models with grid search and k-fold cross-validation, using accuracy as the metric for model selection.'''

		#Create a grid search object with the given pipeline and parameter settings, using 5-fold cross-validation and accuracy as the scoring metric.
		grid_search_item = GridSearchCV(pipeline, self.parameters, cv=5, scoring='accuracy')

		# Fit the grid search object to the training data.
		grid_search = grid_search_item.fit(self.X_train, self.y_train)
		
		# Return the fitted grid search object.
		return grid_search

	def evaluation_methods(self, y_true, y_pred):

		""" Function to calculate TPR, FPR, PPV """

		#get confusion matrix for group 
		CM = confusion_matrix(y_true, y_pred)
		TN = CM[0][0] # Number of true negatives
		FN = CM[1][0] # Number of false negatives
		TP = CM[1][1] # Number of true positives
		FP = CM[0][1] # Number of false positives
        
		## Check for any missing values in the confusion matrix, if there is a missing value, set it to 0
		try:
			TN = CM[0][0]
	
		except IndexError: 
			TN = 0 

		try:	
			FN = CM[1][0]
		except IndexError:
			FN = 0

		try:
			TP = CM[1][1]
		except IndexError:
			TP = 0

		try:
			FP = CM[0][1]
		except IndexError:
			FP = 0

		# Calculate recall: True Positive Rate (TPR)
		if TP + FN != 0:
			TPR = TP / (TP + FN)
		# Set TPR to None if denominator is 0
		else:
			TPR = None

		# Calculate Positive Predictive Value (PPV)
		if FP + TN != 0:
			FPR = FP / (FP + TN)
		
		# Set PPV to None if denominator is 0
		else:
			FPR = None

		# Calculate Positive Predictive Value (PPV)
		if TP + FP != 0:
			PPV = TP / (TP + FP)

		# Set PPV to None if denominator is 0
		else:
			PPV = None

		
        # Return TPR, FPR, PPV as results of the evaluation
		return TPR, FPR, PPV
		
    ## Function to calculate KS test score
	def get_ks_score(self):
		"""
		This function computes the KS score. The KS score measures the distance between the positive and negative class distributions.
		
		It first separates the outcome labels and feature matrix into two groups: positive (label value 1) and negative (label value 0). Then it uses a trained model (a grid search object) to predict the probability of an observation being positive. The prediction probabilities are then separated into positive and negative groups, and the KS score is computed based on these two groups. The ks_2samp function is used to calculate the KS score.
        
		A more detailed explanation of the KS score can be found here: 
		https://towardsdatascience.com/evaluating-classification-models-with-kolmogorov-smirnov-ks-test-e211025f5573 
	    
		"""

		# Merge feature set with outcome in test set
		all_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
		#print(all_test.columns)

		# Separate the feature matrix into two groups according to the outcome labels: positive (1) and negative (0)
		X_1 = all_test[all_test['Dx_OpioidOverdose_0to1_Y'] == 1]
		X_0 = all_test[all_test['Dx_OpioidOverdose_0to1_Y'] == 0]

        # Drop outcome labels in the feature matrix
		X_1.drop(columns=['Dx_OpioidOverdose_0to1_Y'])
		X_0.drop(columns=['Dx_OpioidOverdose_0to1_Y'])
		
		# Divide the outcome labels into two groups: positive, negative
		y_1 = all_test.loc[all_test['Dx_OpioidOverdose_0to1_Y'] == 1].Dx_OpioidOverdose_0to1_Y
		y_0 = all_test.loc[all_test['Dx_OpioidOverdose_0to1_Y'] == 0].Dx_OpioidOverdose_0to1_Y

		# Use the trained model from grid_search to make predictions on the two groups: positive and negative
		y_true_1, y_pred_prob_1 = y_1, grid_search.predict_proba(X_1)
		y_true_0, y_pred_prob_0 = y_0, grid_search.predict_proba(X_0)
        
		#Convert the prediction probability arrays to dataframes. The ks_2samp function requires the input to be dataframe columns.
		y_pred_prob_1_df = pd.DataFrame(y_pred_prob_1)
		y_pred_prob_0_df = pd.DataFrame(y_pred_prob_0)

		#Calculate the KS score using the ks_2samp function. The KS score represents the maximum difference between the cumulative distribution functions of the two groups.
		ks_score = ks_2samp(y_pred_prob_1_df[0], y_pred_prob_0_df[0])
		# print(ks_score)	
		
		return  ks_score



	def get_group_evaluation(self, group_name, y_pred, X_test):
		"""
		This function evaluates performance in different groups using TPR, FPR, and PPV metrics.
		First, we merge feature matrix with outcome and predicted lables

		Inputs:
	 	 - group_name: Name of the column used to separate the feature matrix into groups.
		 - y_pred: Predicted labels for the test set.
		 - X_test: Feature matrix of the test set.

		"""

		# Separate the feature matrix into two groups based on the values of the specified column. 
		# The column is a binary indicator of whether an observation belongs to a specific group 
		X_1 = X_test.loc[X_test[group_name] == 1]
		X_0 = X_test.loc[X_test[group_name] == 0]

		# Merge the feature matrix and the outcome variable for the test set.
		all_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
		
		# Merge the feature matrix, outcome variable, and predicted labels.
		all_test = pd.concat([all_test.reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)], axis=1)
		# Rename the column with the predicted labels.
		all_test = all_test.rename(columns={0: 'prediction'})
		
		#pPartition the outcome variable for each group.
		y_true_1 = all_test.loc[all_test[group_name] == 1].Dx_OpioidOverdose_0to1_Y
		y_true_0 = all_test.loc[all_test[group_name] == 0].Dx_OpioidOverdose_0to1_Y

		# Partition the predicted labels for each group.
		y_pred_1 = all_test.loc[all_test[group_name] == 1].prediction
		y_pred_0= all_test.loc[all_test[group_name] == 0].prediction

		#Get TPR, FPR, and PPV metrics for each group using the evaluation_methods function.
		TPR_1, FPR_1, PPV_1 = self.evaluation_methods(y_true_1, y_pred_1)
		TPR_0, FPR_0, PPV_0 = self.evaluation_methods(y_true_0, y_pred_0)

		return TPR_1, FPR_1, PPV_1, TPR_0, FPR_0, PPV_0 
	

def get_loss_func_weights():
	"""
	Get class weight for loss function according to population and cohort size, see draft secion 3.4 
	"""

	# read data
	data = pd.read_csv(path + "data/simulated_data_big_sample.csv")
	# count of case in population
	count_of_case_pop = len(data.loc[data['Dx_OpioidOverdose_0to1_Y'] == 1]) 
	# count of control in population
	count_of_control_pop = len(data.loc[data['Dx_OpioidOverdose_0to1_Y'] == 0])
	# count of control in cohort
	count_of_control_cohort = count_of_case_pop * 100
	#  W is inverse probability of case or control being drawn from the population in each experiment, therefore, weight of control is:
	W_control = count_of_control_pop/count_of_control_cohort
	W_case = 1
	# normalize W_control and W_case so that they sum up to 1
	W_case_normal = round(W_case/(W_control + W_case), 2)
	W_control_normal = round(W_control/(W_control + W_case), 2)

	return W_case_normal, W_control_normal


#preprocess data, get train, test and outcome
#path = '/Users/luciachen/Dropbox/simulated_data/' 
path = '/home/groups/sherrir/luciachn/simulated_data/'  #path where the data is stored, change it to your path
file_number = 'nonweighted_shuffled' #name of the file, change the file name here

#load experiment parameters
experiment = load_experiment(path + 'experiment.yaml')
#change experiment to weighted loss function
#experiment = load_experiment(path + 'experiment_weighted.yaml')

# Get class weight for loss function according to population and cohort size 
W_case_normal, W_control_normal = get_loss_func_weights()

# modify the weights of the experiment file
# first check if class weight is in the parameter dictionary, if yes, modify the weights accordingly.
if 'clf__classifier__class_weight' in experiment['experiment']['sklearn.linear_model.LogisticRegression']:
	experiment['experiment']['sklearn.linear_model.LogisticRegression']['clf__classifier__class_weight'] = [{0: W_control_normal, 1: W_case_normal}]

#store results
file_exists = os.path.isfile(path + 'results/test_result_{}.csv'.format(file_number)) #remember to create a results folder
f = open(path + 'results/test_result_{}.csv'.format(file_number), 'a')
writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)




#if result file doesn't exist, create file and write column names
if not file_exists:
	writer_top.writerow(['best_scores'] + ['best_parameters'] + ['weighted'] + ['precision'] + ['recall'] + ['f1'] + ['accuracy']+ ['TPR'] + ['FPR'] + ['PPV'] + ['ks_score'] + ['ks_score_p'] + ['time'] + ['model'] +['feature_set']  + ['TPR_nonWhite'] + ['TPR_White'] + ['TPR_backpain'] + ['TPR_nonBackpain'] + ['TPR_neckpain'] + ['TPR_nonNeckpain'] + ['TPR_Neuropathy'] + ['TPR_nonNeuropathy'] +  ['TPR_Fibromyalgia'] +  ['TPR_noFibromyalgia'] + ['TPR_PTSD'] + ['TPR_noPTSD'] +  ['TPR_MDD'] + ['TPR_noMDD'] + ['TPR_Homeless'] + ['TPR_noHomeless'] + ['FPR_nonWhite'] + ['FPR_White'] + ['FPR_backpain'] + ['FPR_nonBackpain'] + ['FPR_neckpain'] + ['FPR_nonNeckpain'] + ['FPR_Neuropathy'] + ['FPR_nonNeuropathy'] +  ['FPR_Fibromyalgia'] +  ['FPR_noFibromyalgia']  + ['FPR_PTSD'] + ['FPR_noPTSD'] +  ['FPR_MDD'] + ['FPR_noMDD'] + ['FPR_Homeless'] + ['FPR_noHomeless'] + ['PPV_nonWhite'] + ['PPV_White'] + ['PPV_backpain'] + ['PPV_nonBackpain'] + ['PPV_neckpain'] + ['PPV_nonNeckpain'] + ['PPV_Neuropathy'] + ['PPV_nonNeuropathy'] +  ['PPV_Fibromyalgia'] +  ['PPV_noFibromyalgia'] + ['PPV_PTSD'] + ['PPV_noPTSD'] +  ['PPV_MDD'] + ['PPV_noMDD'] + ['PPV_Homeless'] + ['PPV_noHomeless'] + ['TPR_male'] + ['TPR_female'] +  ['FPR_male'] + ['FPR_female'] + ['PPV_male'] + ['PPV_female'])
	f.close()
	

#loop through each classifier and parameters defined in experiment.yaml, then we get the classification report of general performance and compare the log loss in each group defined by sensitive attributes

# Set the initial value of i (number of iterations) to 0
i = 0
# Run the experiments 500 times
while i < 1: 
	## Preprocess the data with the given path
	p = PreprocessData(path)
	X_train, X_test, y_train, y_test = p.get_train_test_split()
    
	# Loop through each classifier and its parameters defined in experiment.yaml
	for classifier in experiment['experiment']:
		parameters = experiment['experiment'][classifier]
		
		## Loop through each list of features
		for feature_key, features_list in experiment['features'].items():
			print(features_list)
			
			# Train the model using the given parameters and features list
			train = Training(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, parameters=parameters, features_list=features_list)
			
			# Set up the pipeline for training
			pipeline = train.setup_pipeline(eval(classifier)())
			
			# Perform grid search on the pipeline
			grid_search = train.training_models(pipeline)

			# Get predictions from the trained model
			y_true, y_pred = y_test, grid_search.predict(X_test)
			

			# Get the classification report for the general performance
			report = classification_report(y_true, y_pred, digits=2, output_dict=True)
			ks_score = train.get_ks_score()

			# Evaluate the TPR, FPR, and PPV for the general performance
			TPR, FPR, PPV = train.evaluation_methods(y_true, y_pred)
	

			# Compare group performance for different sensitive attributes
			# gender
			TPR_female, FPR_female, PPV_female, TPR_male, FPR_male, PPV_male= train.get_group_evaluation('sex_female', y_pred, X_test) 
			# race
			TPR_White, FPR_White, PPV_White, TPR_nonWhite, FPR_nonWhite, PPV_nonWhite= train.get_group_evaluation('race_White', y_pred, X_test)
			# back pain
			TPR_backpain, FPR_backpain, PPV_backpain, TPR_nonBackpain, FPR_nonBackpain, PPV_nonBackpain= train.get_group_evaluation('PA_Pain_Back', y_pred, X_test) 
			# neck pain
			TPR_neckpain, FPR_neckpain, PPV_neckpain, TPR_nonNeckpain, FPR_nonNeckpain, PPV_nonNeckpain= train.get_group_evaluation('PA_Pain_Neck_0to1_Y', y_pred, X_test) 
			# neuropathy pain
			TPR_Neuropathy, FPR_Neuropathy, PPV_Neuropathy, TPR_nonNeuropathy, FPR_nonNeuropathy, PPV_nonNeuropathy = train.get_group_evaluation('PA_Pain_Neuropathy_0to1_Y', y_pred, X_test) 
			#Fibromyalgia
			TPR_Fibromyalgia, FPR_Fibromyalgia, PPV_Fibromyalgia, TPR_noFibromyalgia, FPR_noFibromyalgia, PPV_noFibromyalgia= train.get_group_evaluation('PA_Pain_Fibromyalgia_0to1_Y', y_pred, X_test) 
			#PTSD
			TPR_PTSD, FPR_PTSD, PPV_PTSD, TPR_noPTSD, FPR_noPTSD, PPV_noPTSD= train.get_group_evaluation('MH_PTSD_1to2_Y', y_pred, X_test) 
			TPR_MDD, FPR_MDD, PPV_MDD, TPR_noMDD, FPR_noMDD, PPV_noMDD= train.get_group_evaluation('MH_MDD_12m', y_pred, X_test) #MDD
			TPR_Homeless, FPR_Homeless, PPV_Homeless, TPR_noHomeless, FPR_noHomeless, PPV_noHomeless= train.get_group_evaluation('EH_Homeless_1to2_Y', y_pred, X_test) #Homelessness
			

			#Print result row: grid search best score, best parameters, weighted loss function or not, classification report,  experiment time, classifier, selected feature columns, evaluation metrics for groups
			result_row = [[grid_search.best_score_, grid_search.best_params_, 'Nonweighted', report['1']['precision'], report['1']['recall'], report['1']['f1-score'], report['accuracy'], TPR, FPR, PPV, ks_score[0], ks_score[1], str(datetime.datetime.now()), classifier, feature_key,  TPR_nonWhite, TPR_White, TPR_backpain, TPR_nonBackpain, TPR_neckpain, TPR_nonNeckpain, TPR_Neuropathy, TPR_nonNeuropathy, TPR_Fibromyalgia, TPR_noFibromyalgia, TPR_PTSD, TPR_noPTSD, TPR_MDD, TPR_noMDD, TPR_Homeless, TPR_noHomeless, FPR_nonWhite, FPR_White, FPR_backpain, FPR_nonBackpain, FPR_neckpain, FPR_nonNeckpain, FPR_Neuropathy, FPR_nonNeuropathy, FPR_Fibromyalgia, FPR_noFibromyalgia,  FPR_PTSD, FPR_noPTSD, FPR_MDD, FPR_noMDD, FPR_Homeless, FPR_noHomeless, PPV_nonWhite, PPV_White, PPV_backpain, PPV_nonBackpain, PPV_neckpain, PPV_nonNeckpain, PPV_Neuropathy, PPV_nonNeuropathy, PPV_Fibromyalgia, PPV_noFibromyalgia, PPV_PTSD, PPV_noPTSD, PPV_MDD, PPV_noMDD, PPV_Homeless, PPV_noHomeless, TPR_male, TPR_female, FPR_male, FPR_female, PPV_male, PPV_female]]

			# Store test result to file
			f = open(path + 'results/test_result_{}.csv'.format(file_number), 'a')
			writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

			writer_top.writerows(result_row)

			f.close()
			gc.collect() #garbage collection

	i += 1
















