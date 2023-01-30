import dice_ml
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split
import pandas as pd
# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import numpy as np
import random


class TrainCounterFactual:
    def __init__(self):
        self.data_path = "/Users/luciachen/Desktop/simulated_data/simulated_data_big_sample.csv"


    def clean_data(self):
        data_big = pd.read_csv(self.data_path)
        data  = data_big[data_big.columns.drop(list(data_big.filter(regex='relevant')))]
        data = data.drop(['Unnamed: 0'], axis=1)

        return data

    def split_train_test(self, data):

        target2 = data["Dx_OpioidOverdose_0to1_Y"] # outcome variable

        train_dataset2, test_dataset2, _, _ = train_test_split(data,
                                                             target2,
                                                             test_size=0.2,
                                                             random_state=0,
                                                             stratify=target2)

        #data.head(10).to_csv("/Users/luciachen/Desktop/simulated_data/simulated_data_big_sample.csv")
        x_train = train_dataset2.drop(["Dx_OpioidOverdose_0to1_Y"], axis=1)
        y_train = train_dataset2["Dx_OpioidOverdose_0to1_Y"]
        x_test = test_dataset2.drop(["Dx_OpioidOverdose_0to1_Y"], axis=1)
        y_test = test_dataset2["Dx_OpioidOverdose_0to1_Y"]

        return x_train, y_train, x_test, y_test, train_dataset2, test_dataset2


    def train_model(self, x_train, y_train, continuous_features, train_dataset2):
        dice_data = dice_ml.Data(dataframe=train_dataset2, continuous_features= continuous_features, outcome_name='Dx_OpioidOverdose_0to1_Y')

        numerical = continuous_features

        categorical = x_train.columns.difference(numerical)

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        transformations = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical),
                ('cat', categorical_transformer, categorical)])

        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', transformations),
                              ('classifier', LogisticRegression())])
        model = clf.fit(x_train, y_train)

        return model, dice_data

class GetCounterFactual:
    def __init__(self, model):
        self.model = model 


    def getCounterFactuals(self, data, dice_data, proximity, diversity, counterfactual_numbers):

        # Using sklearn backend
        m = dice_ml.Model(model= self.model, backend="sklearn")
        # Using method=random for generating CFs
        exp = dice_ml.Dice(dice_data, m, method="random")


        e1 = exp.generate_counterfactuals(data, total_CFs=counterfactual_numbers, desired_class="opposite", proximity_weight=proximity, diversity_weight=diversity)
        #e1.visualize_as_dataframe(show_only_changes=True)

        return e1, exp


    def get_index(self, x_test, result_df, counterfactual_numbers):
        #append test set index to counterfactual results

        new_id = []
        for i in x_test['case_id']:
            index = 0
            while index < counterfactual_numbers:
                new_id.append(i)
                #print(new_id)
                index += 1
            
            if len(new_id) == result_df.shape[0]:
                break

        return new_id


    def get_counterfactuals_dict(self, x_test, number_of_examples, dice_data, proximity, diversity, counterfactual_numbers):
   
        #store counterfactuals in df
       # case_id = np.arange(len(x_test))

        result_df = pd.DataFrame([])
        result_dict = {}
        for end, index in zip(range(1, x_test.shape[0]), range(0, 10000)):
            start = 0
            e1, exp = self.getCounterFactuals(x_test[start: end], dice_data, proximity, diversity, counterfactual_numbers)
           # result_df = result_df.append(e1.cf_examples_list[0].final_cfs_df)
            result_d = e1.cf_examples_list[0].final_cfs_df.to_dict()
            
            #create a unique id for each counterfactual
            random_number = (random.randint(0,10000000000))
            unique_index = str(index) + str(random_number)

            result_dict[unique_index] = {}
            result_dict[unique_index]['result'] = result_d
            result_dict[unique_index]['result']['proximity'] = proximity
            result_dict[unique_index]['result']['diversity'] = diversity

            start = start + 1
            if end == number_of_examples:
                break

        return exp, e1, result_dict


    def get_counterfactuals_df(self, result_dict, counter_factual_numbers):

        result_df = pd.DataFrame([])
       
        for key in result_dict.keys():
            df = pd.DataFrame.from_dict(result_dict[key]['result'])
            result_df = result_df.append(df)
           

        x_test['case_id'] = np.arange(len(x_test))

        index = self.get_index(x_test, result_df, counter_factual_numbers) 

        result_df['index'] = index

        return result_df


continuous_features = ['Ux_OP_PoC_ED_30days', 'Ux_OP_PoC_ED_91days', 'Ux_OP_PoC_ED_365days','Ux_OP_PoC_ED_730days', 'Ux_OP_PoC_UC_30days', 'Ux_OP_PoC_UC_91days','Ux_OP_PoC_UC_182days', 'Ux_OP_PoC_UC_365days', 'Ux_OP_PoC_UC_730days','MOPROB_AllAppt', 'MOPROB_AllAppt_30', 'MOPROB_AllAppt_60', 'MOPROB_AllAppt_90', 'Ux_OP_PoC_TobaccoCessation_30day','Ux_OP_PoC_TobaccoCessation_91day', 'Ux_OP_PoC_TobaccoCessation_182day','Ux_OP_PoC_TobaccoCessation_365day','Ux_OP_PoC_TobaccoCessation_730day', 'TotalNumOfAppt_182days','TotalNumOfAppt_91days', 'TotalNumOfAppt_365days', 'TotalNumOfAppt_730days', 'Ux_OP_MHOC_HBPC_30days','Ux_OP_MHOC_HBPC_91days', 'Ux_OP_MHOC_HBPC_182days','Ux_OP_MHOC_HBPC_365days', 'Ux_OP_MHOC_HBPC_730days', 'Rx_Medd',
                'Rx_Medd_10m', 'Rx_Medd_11m', 'Rx_Medd_1m', 'Rx_Medd_2m', 'Rx_Medd_3m', 'Rx_Medd_4m', 'Rx_Medd_5m', 'Rx_Medd_6m', 'Rx_Medd_7m', 'Rx_Medd_8m','Rx_Medd_9m']

t = TrainCounterFactual()
data = t.clean_data()
data = data.sample(frac=0.2, replace=True, random_state=1)
data = shuffle(data)
x_train, y_train, x_test, y_test,  train_dataset2, test_dataset2 = t.split_train_test(data)
model, dice_data = t.train_model(x_train, y_train, continuous_features, train_dataset2) #logisitics regression

g = GetCounterFactual(model)
proximity_weights = [1, 1.5]
diversity_weights = [0.5, 1]

result_dict_all = {}
counterfactual_numbers = 2
for prox_w in proximity_weights:
    for diver_w in diversity_weights:
        print(prox_w, diver_w)
        e1, exp, result_dict = g.get_counterfactuals_dict(x_test, 5, dice_data, prox_w, diver_w, counterfactual_numbers)

        #append result to big result dictionary 
        result_dict_all.update(result_dict)
        print('length of result dict', len(result_dict_all))

result_df = g.get_counterfactuals_df(result_dict_all, counterfactual_numbers)


# Save generated counterfactual examples to disk
result_df.to_csv(path_or_buf='/Users/luciachen/Desktop/simulated_data/counterfactuals/counterfactuals1.csv', index=False)


test_dataset2[0:10].to_csv('/Users/luciachen/Desktop/simulated_data/counterfactuals/testset.csv')

# #get local feature importance, feature importance for one point

# x_test = x_test.drop(['case_id'], axis=1)
# imp = exp.local_feature_importance(x_test[0:1], total_CFs=10)
# print(imp.local_importance)

# #get global 
# query_instances = x_test[0:20]
# imp = exp.global_feature_importance(query_instances)
# print(imp.summary_importance)

# # plot decision boundries 
# ax = plot_decision_regions(x_train, y_train, clf=model, legend=2)
# ax.scatter(*e1
# plt.show()











































