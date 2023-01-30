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
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import log_loss

import csv
import gc
import datetime
import glob
import os
import random

#write down what does each trunk of code do 

def load_experiment(path_to_experiment):
	#load experiment 
	data = yaml.safe_load(open(path_to_experiment))
	return data



class PreprocessData:
	def __init__(self, path):
		self.data_path = path
		self.label = 'Dx_OpioidOverdose_0to1_Y'


	# def get_data(self):
	# 	"""Read data
	# 	output: pandas dataframe
	# 	"""

	# 	data = pd.read_csv(self.data_path + "cohorts_data.csv")

	# 	return data


	def get_cohorts(self):
		'''
		Get case-control 
		'''
		data = pd.read_csv(self.data_path + "simulated_data_big_sample.csv")


		positive = data.loc[data['Dx_OpioidOverdose_0to1_Y'] == 1]
		negative = data.loc[data['Dx_OpioidOverdose_0to1_Y'] == 0]

		negative_cohort = negative.sample(n=positive.shape[0]*100, random_state=random.randint(0,10000))
		case_control_cohort = pd.concat([positive,negative_cohort])

		return case_control_cohort


	def pre_process(self): 
		"""get labels
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

		X, y= self.pre_process()
		# get 10% holdout set for testing
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 300, stratify = y)

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
	def __init__(self, X_train, X_test, y_train, y_test, parameters, features_list):
		self.X_train = X_train #train feature
		self.y_train = y_train #train outcome
		self.X_test = X_test #test feature
		self.y_test = y_test #test outcome
		self.parameters = parameters #parameters, see experiment.yaml
		self.path = "/Users/luciachen/Desktop/simulated_data/" #path for the simulated dataset
		self.label = 'Dx_OpioidOverdose_0to1_Y' #outcome 
		self.features_list = features_list #list of features include in training, see experiment.yaml


	def get_feature_names(self):
		"""Select all the merged features. """

		fea_list = []
		for fea in self.features_list: #select column names with keywords in dict
			f_list = [i for i in self.X_train.columns if fea in i]
			fea_list.append(f_list)

		#flatten a list
		flat = [x for sublist in fea_list for x in sublist]
		#convert to transformer object
		return flat



	def setup_pipeline(self, classifier):
		'''set up pipeline'''
		#select features from experiment file
		features_col = self.get_feature_names()


		pipeline = Pipeline([

			
		# generate count vect features
		  # # select other features, feature sets are defines in the yaml file
		('other_features', Pipeline([

				('selector', ColumnSelector(columns=features_col)),
				('impute', SimpleImputer(strategy='mean')),# impute nan with mean
			 ])),


		('clf', Pipeline([
			   # ('impute', SimpleImputer(strategy='mean')), #impute nan with mean
			   ('scale', StandardScaler(with_mean=False)),  # scale features
				('classifier', classifier),  # classifier
		   
				 ])),
		])
		return pipeline

	def training_models(self, pipeline):
		'''train models with grid search'''
		grid_search_item = GridSearchCV(pipeline, self.parameters, cv=5, scoring='accuracy')
		grid_search = grid_search_item.fit(self.X_train, self.y_train)
		
		return grid_search


	def get_group_loss(self, group_name):

		X_1 = X_test.loc[X_test[group_name] == 1]
		X_0 = X_test.loc[X_test[group_name] == 0]

		all_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

		y_1 = all_test.loc[all_test[group_name] == 1].Dx_OpioidOverdose_0to1_Y
		y_0 = all_test.loc[all_test[group_name] == 0].Dx_OpioidOverdose_0to1_Y

		y_true_1, y_pred_prob_1 = y_1, grid_search.predict_proba(X_1)
		y_true_0, y_pred_prob_0 = y_0, grid_search.predict_proba(X_0)
		log_loss_1 = sklearn.metrics.log_loss(y_true_1, y_pred_prob_1)
		log_loss_0 = sklearn.metrics.log_loss(y_true_0, y_pred_prob_0)

		return log_loss_1, log_loss_0



#preprocess data, get train, test and outcome
path = '/Users/luciachen/Desktop/simulated_data/' #change path in here

p = PreprocessData(path)
X_train, X_test, y_train, y_test = p.get_train_test_split()


experiment = load_experiment(path + 'experiment.yaml')

#store results
file_exists = os.path.isfile(path + 'results/test_result2.csv') #remember to create a results folder
f = open(path + 'results/test_result2.csv', 'a')
writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
if not file_exists:
    writer_top.writerow(['best_scores'] + ['best_parameters'] + ['report'] + ['log_loss'] + ['time'] + ['model'] +['feature_set'] + ['female_loss'] +['male_loss'] + ['nonWhite_loss'] + ['White_loss'] + ['backpain_loss'] + ['nonBackpain_loss'] + ['neckpain_loss'] + ['nonNeckpain_loss'] + ['Neuropathy_loss'] + ['nonNeuropathy_loss'])
    f.close()
    

for classifier in experiment['experiment']:
	parameters = experiment['experiment'][classifier]
    
    #loop through lists of features
	for key, features_list in experiment['features'].items():
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

		#get classification report
		report = classification_report(y_true, y_pred, digits=2)
		log_loss = log_loss(y_true, y_pred_prob)

		#sensitive attributes "female", "white", "Dx_Pain_Back", "Dx_Pain_Neuropathy_0to1_Y", "Dx_Pain_Neck_0to1_Y
		female_loss, male_loss = train.get_group_loss('sex_female')
		nonWhite_loss, White_loss = train.get_group_loss('race_NonWhite')
		backpain_loss, nonBackpain_loss = train.get_group_loss('Dx_Pain_Back')
		neckpain_loss, nonNeckpain_loss = train.get_group_loss('Dx_Pain_Neck_0to1_Y')
		Neuropathy_loss, nonNeuropathy_loss = train.get_group_loss('Dx_Pain_Neuropathy_0to1_Y')


		#define a row of results we want to store
		result_row = [[grid_search.best_score_, grid_search.best_params_, report, log_loss, str(datetime.datetime.now()), classifier, features_list, female_loss, male_loss, nonWhite_loss, White_loss, backpain_loss, nonBackpain_loss, neckpain_loss, nonNeckpain_loss, Neuropathy_loss, nonNeuropathy_loss]]

		# store test result
		f = open(path + 'results/test_result2.csv', 'a')
		writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

		writer_top.writerows(result_row)

		f.close()
		gc.collect()

#append logloss with feature matrix














