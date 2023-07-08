# NarxCare Project

This Repo contains all the scripts for the auditing NarxCare ORS project.

Require Packages:
R: simstudy, ggplot2, Envstats, truncnorm, psych, ids, data.table, scales, dplyr, tidyr, cowplot, tibble, forcats, dplyr
Python: pandas, numpy, sklearn, re, typing (for type setting), ruamel (for yaml file), scipy, csv, gc, datetime, glob, os, random 

## Source (Folder)

### R (Folder)
1. generate_simulated_data.Rmd: script that generates the simulated datasets, including the shifted dataset.
2. simulated_data_functions.R: functions used in generate_simulated_data.Rmd.
3. plot_result.Rmd: script that generates all the result plots on the manuscript.

### Python (Folder)
#### paramter files
1. experiment_weighted.yaml: setting parameters for model with weighted loss function
2. experiment.yaml: setting parameters 

#### model files
1. training_models_weighted_shifted.py: model with weighted loss function and shifted data
2. training_models_weighted.py: model with weighted loss function
3. training_models.py: model with non-weighted loss function

## Simulated Data (Folder, in a separate dropbox link):

## How to run the model file:

training_models.py 
1. change path to your local machine: line 328 path = <yourPath>
2. change experiment file path line: 333
3. create a results folder under <yourPath> to store the test result, an example of the model result is here https://www.dropbox.com/s/hrujn6zesf9ath9/test_result_weighted.csv?dl=0

Repeat the same procedures with training_models_weighted.py
