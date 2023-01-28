# NarxCare-project
This project is about auditing the NarxCare ORS


## Data:

simulated_data_big_sample.csv: 2 million simulated data
cohorts_data.csv: simulated data with nested case-control sampling 
simulated_data_big_sample_reduced_mean.csv: REDD mean reduced 5%

## Experiments:

1. experiment 1: algorithm approximation

We use logisitic regression, SVM and random forest. We can set the class weights for the nest case-control study in experiment.yaml

training_models.py: pipeline for model training, change self.data_path to run on server, parameters can be set in experiment.yaml. This pipeline generates a result file named "test_result2.csv"

experiment.yaml: set parameters for the pipeline. Please Uncomment each block to use different feature sets and parameters


2. experiment 2: adjusted weight for nested case-control sampling

Run pipeline with the weighted algorithm
the weighted algorithm can be achieved by changing the class weight in experiment.yaml


3. experiment 3: datashift. 
Run pipeline with the shifted data


4. experiment 4: Group fairness
Group loss result is printed on "test_result2.csv"

