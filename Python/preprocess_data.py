import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.model_selection import train_test_split



class PreprocessData:

	def __init__(self):
		self.data_path = "/Users/luciachen/Desktop/simulated_data/"
		self.label = 'Dx_OpioidOverdose_0to1_Y' #outcome column 


	def get_data(self):
		"""Read data"""

		data = pd.read_csv(self.data_path + "cohorts_data.csv")

		return data


	def pre_process(self): 
		"""get labels"""
		data = self.get_data()

		X = data.drop(columns=[self.label, 'Unnamed: 0', 'X', 'MVIPersonSID'])
		y = data[self.label]

		return X, y


	def get_train_test_split(self):
		''' split train test 8:2, stratify splitting'''

		X, y= self.pre_process()
		# get 10% holdout set for testing
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 300, stratify = y)

		return X_train, X_test, y_train, y_test



































