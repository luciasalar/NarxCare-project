import dice_ml
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split
import pandas as pd
# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.utils import shuffle
from sklearn.metrics import classification_report


data_path = "/Users/luciachen/Desktop/simulated_data/simulated_data_big_sample.csv"
data_big = pd.read_csv(data_path)
data  = data_big[data_big.columns.drop(list(data_big.filter(regex='relevant')))]
data = data.drop(['Unnamed: 0'], axis=1)
# data = data[['MOPROB_AllAppt', 'MOPROB_AllAppt_30', 'MOPROB_AllAppt_60',
#        'MOPROB_AllAppt_90', 'Ux_OP_PoC_ED_30days', 'Ux_OP_PoC_ED_91days', 'Ux_OP_PoC_ED_365days',
#        'Ux_OP_PoC_ED_730days', 'Ux_OP_PoC_UC_30days', 'Ux_OP_PoC_UC_91days',
#        'Ux_OP_PoC_UC_182days', 'Ux_OP_PoC_UC_365days', 'Ux_OP_PoC_UC_730days', 'Ux_OP_PoC_TobaccoCessation_30day',
#        'Ux_OP_PoC_TobaccoCessation_91day', 'Ux_OP_PoC_TobaccoCessation_182day',
#        'Ux_OP_PoC_TobaccoCessation_365day',
#        'Ux_OP_PoC_TobaccoCessation_730day', 'Dx_Pain_Urogenital_0to1_Y',
#        'Dx_Pain_Neuropathy_0to1_Y', 'Dx_Pain_Back', 'Dx_Pain_Neck_0to1_Y', 'Dx_OpioidOverdose_0to1_Y']]
data = data.sample(frac=0.2, replace=True, random_state=1)
data = shuffle(data)
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

x_train = x_train.to_numpy()

clf2 = RandomForestClassifier)
model = clf2.fit(x_train, y_train)

y_pred = clf2.predict(x_test)
classification_report(y_test, y_pred, digits=2)

x_ref2 = x_test.to_numpy()[2]


print('True label:', y_test.to_numpy()[2])
print('Predicted label:',  clf2.predict(x_ref2.reshape(1, -1))[0])

print('Predicted probas:', clf2.predict_proba(x_ref2.reshape(1, -1)))
print('Predicted probability for label 0:', model.predict_proba(x_ref2.reshape(1, -1))[0][0])


res = create_counterfactual(x_reference = x_ref2, 
                            y_desired=1, 
                            model=clf2, 
                            X_dataset=x_train,
                            y_desired_proba=1,
                            lammbda=1, #  hyperparameter
                            random_seed=123)


print('Predictions for counterfactual:\n')
print('Predicted label:', clf2.predict(res.reshape(1, -1))[0])
print('Predicted probas:', clf2.predict_proba(res.reshape(1, -1)))







