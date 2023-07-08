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
from sklearn.metrics import log_loss
from scipy.stats import ks_2samp

import csv
import gc
import datetime
import glob
import os
import random

#This script returns the prediction probability, true outcome and features in the test set

def load_experiment(path_to_experiment):
	#load experiment 
	data = yaml.safe_load(open(path_to_experiment))
	return data



class PreprocessData:
	def __init__(self, path):
		self.data_path = path
		self.label = 'Dx_OpioidOverdose_0to1_Y' #outcome label: opioid overdose in the past one year


	def get_cohorts(self):
		'''
		Get case-control cohort 
		'''
		data = pd.read_csv(self.data_path + "simulated_data_big_sample.csv")

		positive = data.loc[data['Dx_OpioidOverdose_0to1_Y'] == 1]
		negative = data.loc[data['Dx_OpioidOverdose_0to1_Y'] == 0]
		
		#select 100 negative observations for every positive observations 
		negative_cohort = negative.sample(n=positive.shape[0]*100, random_state=random.randint(0,10000))

		#combine all the cohorts
		case_control_cohort = pd.concat([positive,negative_cohort])

		return case_control_cohort


	def pre_process(self): 
		"""
		create feature matrix and outcome variable
		output: feature matrix X, outcome y as a vector
		"""
		data = self.get_cohorts()

		X = data.drop(columns=[self.label])
		y = data[self.label]

		return X, y

	def get_train_test_split(self):
		''' split train test 8:2, stratify splittin
		output: X train / test feature matrix, y train/test outcome
		'''
		#get feature matrix and outcome variable
		X, y= self.pre_process()
		# get 20% holdout set for testing
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y)
		
		return X_train, X_test, y_train, y_test



class ColumnSelector:
	'''feature selector for pipline (pandas df format)
	Here we select the features used in the pipeline, feature groups are listed in experiment.yaml

	 '''
	def __init__(self, columns):
		self.columns = columns #columns to be selected for training

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		assert isinstance(X, pd.DataFrame)

		#if data doesn't contain the specified column, ignore the error
		try:
			return X[self.columns]
		except KeyError:
			cols_error = list(set(self.columns) - set(X.columns))
			raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

	def get_feature_names(self):
		return self.columns.tolis


class Training:
	"""
	the training class includes a training pipeline, the training pipeline contains processing feature sets selected in experiment.yaml, feature processing includes normalization and imputation of the features. The model was trained with k fold cross validation, finally we compare the log loss between groups defined by sensitive attributes

	"""
	def __init__(self, X_train, X_test, y_train, y_test, parameters, features_list):

		#define variables pass in the functions within the class
		self.X_train = X_train #train feature
		self.y_train = y_train #train outcome
		self.X_test = X_test #test feature
		self.y_test = y_test #test outcome
		self.parameters = parameters #parameters, see experiment.yaml
		self.path = "/Users/luciachen/Dropbox/simulated_data/" #path for the simulated dataset
		self.label = 'Dx_OpioidOverdose_0to1_Y'#outcome 
		self.features_list = features_list #list of features included in training, see experiment.yaml


	def get_feature_names(self):
		"""Select features for training. """

		fea_list = []
		for fea in self.features_list: 
			#select feature column in train feature matrix, feature columns are defined in experiment.yaml
			f_list = [i for i in self.X_train.columns if fea in i]
			fea_list.append(f_list)

		#flatten the feature list
		flat = [x for sublist in fea_list for x in sublist]

		return flat



	def setup_pipeline(self, classifier):
		'''set up pipeline'''

		#select features from experiment file
		features_col = self.get_feature_names()

		#pipeline
		pipeline = Pipeline([

		('other_features', Pipeline([
				
				#insert feature sets in the pipeline
				('selector', ColumnSelector(columns=features_col)),
				('impute', SimpleImputer(strategy='mean')),# impute nan with mean
			 ])),

		('clf', Pipeline([
			
			   ('scale', StandardScaler(with_mean=False)),  # scale features
				('classifier', classifier),  # classifier, classifer is defined in experiment.yaml
		   
				 ])),
		])
		return pipeline

	def training_models(self, pipeline):

		'''train models with grid search, k fold cross validation, using accuracy to select the best model'''
		grid_search_item = GridSearchCV(pipeline, self.parameters, cv=5, scoring='accuracy')
		grid_search = grid_search_item.fit(self.X_train, self.y_train)
		
		return grid_search

	def evaluation_methods(self, y_true, y_pred):
		"""compute TPR, FPR, PPV """

		#get confusion matrix for group 
		CM = confusion_matrix(y_true, y_pred)
		TN = CM[0][0]
		FN = CM[1][0]
		TP = CM[1][1]
		FP = CM[0][1]

		#calculate recall: TPR
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

		#calculate recall: TPR
		if TP + FN != 0:
			TPR = TP / (TP + FN)
		else:
			TPR = None

		#calculate FPR
		if FP + TN != 0:
			FPR = FP / (FP + TN)
		else:
			FPR = None

		#calculate PPV
		if TP + FP != 0:
			PPV = TP / (TP + FP)
		else:
			PPV = None

		return TPR, FPR, PPV
		

	def get_group_loss(self, group_name):
		"""compare the log loss in different groups
		group_name: groups for comparison

		"""

		X_1 = X_test.loc[X_test[group_name] == 1]
		X_0 = X_test.loc[X_test[group_name] == 0]

		#merge feature set with outcome in test set
		all_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
		
		#subset outcome according to group name
		y_1 = all_test.loc[all_test[group_name] == 1].Dx_OpioidOverdose_0to1_Y
		y_0 = all_test.loc[all_test[group_name] == 0].Dx_OpioidOverdose_0to1_Y


		#used the trained model to make predictions on different groups, y_pred_prob_1: prediciton, y_true_1: outcome
		y_true_1, y_pred_prob_1 = y_1, grid_search.predict_proba(X_1)
		y_true_0, y_pred_prob_0 = y_0, grid_search.predict_proba(X_0)

		#calculate the log loss in each group
		log_loss_1 = sklearn.metrics.log_loss(y_true_1, y_pred_prob_1)
		log_loss_0 = sklearn.metrics.log_loss(y_true_0, y_pred_prob_0)
		
		
		return log_loss_1, log_loss_0

	def get_ks_score(self):
		"""get ks test score
		measure the distance between the positive and negative class distributions,
		https://towardsdatascience.com/evaluating-classification-models-with-kolmogorov-smirnov-ks-test-e211025f5573

		"""
		#X_1 = X_test.loc[X_test[group_name] == 1]
		#X_0 = X_test.loc[X_test[group_name] == 0]

		#merge feature set with outcome in test set
		all_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
		print(all_test.columns)

		#split X_test according to outcome label
		X_1 = all_test[all_test['Dx_OpioidOverdose_0to1_Y'] == 1]
		X_1.drop(columns=['Dx_OpioidOverdose_0to1_Y'])
		X_0 = all_test[all_test['Dx_OpioidOverdose_0to1_Y'] == 0]
		X_0.drop(columns=['Dx_OpioidOverdose_0to1_Y'])
		
		# #subset outcome according to group name
		y_1 = all_test.loc[all_test['Dx_OpioidOverdose_0to1_Y'] == 1].Dx_OpioidOverdose_0to1_Y
		y_0 = all_test.loc[all_test['Dx_OpioidOverdose_0to1_Y'] == 0].Dx_OpioidOverdose_0to1_Y


		# #used the trained model to make predictions on different groups, y_pred_prob_1: prediciton, y_true_1: outcome
		y_true_1, y_pred_prob_1 = y_1, grid_search.predict_proba(X_1)
		y_true_0, y_pred_prob_0 = y_0, grid_search.predict_proba(X_0)

		y_pred_prob_1_df = pd.DataFrame(y_pred_prob_1)
		y_pred_prob_0_df = pd.DataFrame(y_pred_prob_0)

		# #calculate ks score
		ks_score = ks_2samp(y_pred_prob_1_df[0], y_pred_prob_0_df[0])
		# print(ks_score)	
		
		return  ks_score
	
	def get_prediction_prob_y_features(self):
		"""get ks test score
		measure the distance between the positive and negative class distributions,
		https://towardsdatascience.com/evaluating-classification-models-with-kolmogorov-smirnov-ks-test-e211025f5573

		"""
		#merge feature set with outcome in test set
		all_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
		print(all_test.columns)

		
		X_test.drop(columns=['Dx_OpioidOverdose_0to1_Y'])

		# #used the trained model to make predictions on different groups, y_pred_prob_1: prediciton, y_true_1: outcome
		y_true, y_pred_prob = y_test, grid_search.predict_proba(X)
		
		
		return  y_true, y_pred_prob, X_test



	def get_group_evaluation(self, group_name, y_pred, X_test):
		"""compare performance in different groups
		group_name: groups for comparison

		"""
		X_1 = X_test.loc[X_test[group_name] == 1]
		X_0 = X_test.loc[X_test[group_name] == 0]

		#merge feature set with outcome in test set
		all_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
		
		#merge feature set with outcome and predicted outcome
		all_test = pd.concat([all_test.reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)], axis=1)
		all_test = all_test.rename(columns={0: 'prediction'})
		
		#subset outcome according to group name
		y_true_1 = all_test.loc[all_test[group_name] == 1].Dx_OpioidOverdose_0to1_Y
		y_true_0 = all_test.loc[all_test[group_name] == 0].Dx_OpioidOverdose_0to1_Y

		#subset prediction according to group name
		y_pred_1 = all_test.loc[all_test[group_name] == 1].prediction
		y_pred_0= all_test.loc[all_test[group_name] == 0].prediction


		#get confusion matrix for group 
		TPR_1, FPR_1, PPV_1 = self.evaluation_methods(y_true_1, y_pred_1)
		TPR_0, FPR_0, PPV_0 = self.evaluation_methods(y_true_0, y_pred_0)

		return TPR_1, FPR_1, PPV_1, TPR_0, FPR_0, PPV_0 
	



#preprocess data, get train, test and outcome
path = '/Users/luciachen/Dropbox/simulated_data/' #change path in here
#path = '/home/groups/sherrir/luciachn/simulated_data/'
file_number = 'weighted'


experiment = load_experiment(path + 'source/Python/experiment_weighted.yaml')

#store results
file_exists = os.path.isfile(path + 'results/test_result_{}.csv'.format(file_number)) #remember to create a results folder
f = open(path + 'results/test_result_{}.csv'.format(file_number), 'a')
writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

#if result file doesn't exist, create file and write column names
if not file_exists:
	writer_top.writerow(['best_scores'] + ['best_parameters'] + ['weighted'] + ['precision'] + ['recall'] + ['f1'] + ['accuracy']+ ['TPR'] + ['FPR'] + ['PPV'] + ['ks_score'] + ['ks_score_p'] + ['time'] + ['model'] +['feature_set']  + ['TPR_nonWhite'] + ['TPR_White'] + ['TPR_backpain'] + ['TPR_nonBackpain'] + ['TPR_neckpain'] + ['TPR_nonNeckpain'] + ['TPR_Neuropathy'] + ['TPR_nonNeuropathy'] +  ['TPR_Fibromyalgia'] +  ['TPR_noFibromyalgia'] + ['TPR_PTSD'] + ['TPR_noPTSD'] +  ['TPR_MDD'] + ['TPR_noMDD'] + ['TPR_Homeless'] + ['TPR_noHomeless'] + ['FPR_nonWhite'] + ['FPR_White'] + ['FPR_backpain'] + ['FPR_nonBackpain'] + ['FPR_neckpain'] + ['FPR_nonNeckpain'] + ['FPR_Neuropathy'] + ['FPR_nonNeuropathy'] +  ['FPR_Fibromyalgia'] +  ['FPR_noFibromyalgia']  + ['FPR_PTSD'] + ['FPR_noPTSD'] +  ['FPR_MDD'] + ['FPR_noMDD'] + ['FPR_Homeless'] + ['FPR_noHomeless'] + ['PPV_nonWhite'] + ['PPV_White'] + ['PPV_backpain'] + ['PPV_nonBackpain'] + ['PPV_neckpain'] + ['PPV_nonNeckpain'] + ['PPV_Neuropathy'] + ['PPV_nonNeuropathy'] +  ['PPV_Fibromyalgia'] +  ['PPV_noFibromyalgia'] + ['PPV_PTSD'] + ['PPV_noPTSD'] +  ['PPV_MDD'] + ['PPV_noMDD'] + ['PPV_Homeless'] + ['PPV_noHomeless'] + ['TPR_male'] + ['TPR_female'] +  ['FPR_male'] + ['FPR_female'] + ['PPV_male'] + ['PPV_female'])
	f.close()
	

#loop through each classifier and parameters defined in experiment.yaml, then we get the classification report of general performance and compare the log loss in each group defined by sensitive attributes
i = 0
while i < 1: #set the number of times we run the experiments
	p = PreprocessData(path)
	X_train, X_test, y_train, y_test = p.get_train_test_split()

	for classifier in experiment['experiment']:
		parameters = experiment['experiment'][classifier]
		
		#loop through lists of features
		for feature_key, features_list in experiment['features'].items():
			print(features_list)
			
			#train model 
			train = Training(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, parameters=parameters, features_list=features_list)
			
			#set up pipeline
			pipeline = train.setup_pipeline(eval(classifier)())
			
			#grid search
			grid_search = train.training_models(pipeline)

			#get predictions
			y_true, y_pred = y_test, grid_search.predict(X_test)
			
			#get prediction probabilities
			y_true, y_pred_prob = y_test, grid_search.predict_proba(X_test)
			
			#table for plotting results
			prediction_prob = pd.DataFrame(y_pred_prob, columns = ['negative_class','positive_class'])
			y_features = pd.concat([y_true, X_test], axis=1)
			y_features = y_features.reset_index(drop=True)
			prob_df = pd.concat([prediction_prob, y_features], axis=1)
			prob_df.to_csv(path + "results/prob_plot_weighted.csv")
	


			#get classification report
			report = classification_report(y_true, y_pred, digits=2, output_dict=True)
	

			#log_loss = log_loss(y_true, y_pred_prob)
			ks_score = train.get_ks_score()

			#get TPR, FPR and PPV for the general performance
			TPR, FPR, PPV = train.evaluation_methods(y_true, y_pred)
			
	

			#Here we compare group performance 
			TPR_female, FPR_female, PPV_female, TPR_male, FPR_male, PPV_male= train.get_group_evaluation('sex_female', y_pred, X_test) #pred and outcomes are all 0 for female
			TPR_White, FPR_White, PPV_White, TPR_nonWhite, FPR_nonWhite, PPV_nonWhite= train.get_group_evaluation('race_White', y_pred, X_test)
			TPR_backpain, FPR_backpain, PPV_backpain, TPR_nonBackpain, FPR_nonBackpain, PPV_nonBackpain= train.get_group_evaluation('PA_Pain_Back', y_pred, X_test) #back pain
			TPR_neckpain, FPR_neckpain, PPV_neckpain, TPR_nonNeckpain, FPR_nonNeckpain, PPV_nonNeckpain= train.get_group_evaluation('PA_Pain_Neck_0to1_Y', y_pred, X_test) #neck pain
			TPR_Neuropathy, FPR_Neuropathy, PPV_Neuropathy, TPR_nonNeuropathy, FPR_nonNeuropathy, PPV_nonNeuropathy = train.get_group_evaluation('PA_Pain_Neuropathy_0to1_Y', y_pred, X_test) #neurological pain
			TPR_Fibromyalgia, FPR_Fibromyalgia, PPV_Fibromyalgia, TPR_noFibromyalgia, FPR_noFibromyalgia, PPV_noFibromyalgia= train.get_group_evaluation('PA_Pain_Fibromyalgia_0to1_Y', y_pred, X_test) #Fibromyalgia
			#TPR_Endometriosis, FPR_Endometriosis, PPV_Endometriosis, TPR_noEndometriosis, FPR_noEndometriosis, PPV_noEndometriosis= train.get_group_evaluation('Dx_Endometriosis_0to1_Y')
			TPR_PTSD, FPR_PTSD, PPV_PTSD, TPR_noPTSD, FPR_noPTSD, PPV_noPTSD= train.get_group_evaluation('MH_PTSD_1to2_Y', y_pred, X_test) #PTSD
			TPR_MDD, FPR_MDD, PPV_MDD, TPR_noMDD, FPR_noMDD, PPV_noMDD= train.get_group_evaluation('MH_MDD_12m', y_pred, X_test) #MDD
			TPR_Homeless, FPR_Homeless, PPV_Homeless, TPR_noHomeless, FPR_noHomeless, PPV_noHomeless= train.get_group_evaluation('EH_Homeless_1to2_Y', y_pred, X_test) #Homelessness

			

			#combine the result columns: grid search best score, best parameters, weighted loss (TRUE/FALSE) classification report, log loss, experiment time, classifier, feature set, log loss of sensitive groups
			result_row = [[grid_search.best_score_, grid_search.best_params_, 'Nonweighted', report['1']['precision'], report['1']['recall'], report['1']['f1-score'], report['accuracy'], TPR, FPR, PPV, ks_score[0], ks_score[1], str(datetime.datetime.now()), classifier, feature_key,  TPR_nonWhite, TPR_White, TPR_backpain, TPR_nonBackpain, TPR_neckpain, TPR_nonNeckpain, TPR_Neuropathy, TPR_nonNeuropathy, TPR_Fibromyalgia, TPR_noFibromyalgia, TPR_PTSD, TPR_noPTSD, TPR_MDD, TPR_noMDD, TPR_Homeless, TPR_noHomeless, FPR_nonWhite, FPR_White, FPR_backpain, FPR_nonBackpain, FPR_neckpain, FPR_nonNeckpain, FPR_Neuropathy, FPR_nonNeuropathy, FPR_Fibromyalgia, FPR_noFibromyalgia,  FPR_PTSD, FPR_noPTSD, FPR_MDD, FPR_noMDD, FPR_Homeless, FPR_noHomeless, PPV_nonWhite, PPV_White, PPV_backpain, PPV_nonBackpain, PPV_neckpain, PPV_nonNeckpain, PPV_Neuropathy, PPV_nonNeuropathy, PPV_Fibromyalgia, PPV_noFibromyalgia, PPV_PTSD, PPV_noPTSD, PPV_MDD, PPV_noMDD, PPV_Homeless, PPV_noHomeless, TPR_male, TPR_female, FPR_male, FPR_female, PPV_male, PPV_female]]

			# # store test result
			f = open(path + 'results/test_result_{}.csv'.format(file_number), 'a')
			writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

			writer_top.writerows(result_row)

			f.close()
			gc.collect()

	i += 1
















