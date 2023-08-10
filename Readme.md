# NarxCare Project

This repository contains all the scripts for the auditing NarxCare ORS project.

## Required Packages
R:
- simstudy
- ggplot2
- Envstats
- truncnorm
- psych
- ids
- data.table
- scales
- dplyr
- tidyr
- cowplot
- tibble
- forcats
- dplyr

Python:
- pandas
- numpy
- sklearn
- re
- typing (for type setting)
- ruamel (for yaml file)
- scipy
- csv
- gc
- datetime
- glob
- os
- random 

## Source (Folder)

### R (Folder)
- `generate_simulated_data.Rmd`: script that generates the simulated datasets, including the shifted dataset.
- `simulated_data_functions.R`: functions used in `generate_simulated_data.Rmd`.
- `plot_result.Rmd`: script that generates all the result plots on the manuscript.

### Python (Folder)
#### Paramter Files
- `experiment_weighted.yaml`: setting parameters for model with weighted loss function
- `experiment.yaml`: setting parameters 

#### Model Files
- `training_models.py`:  training model

## Simulated Data (Folder, in a separate dropbox link):
We simulated a population sample consisting of 2 million observations, 52 independent variables, and one dependent (outcome) variable (Opioid Overdose incident in the past year). 


### Shifted datasets
- `simulated_data_big_sample.csv`: Data set
- `simulated_data_big_sample_shuffled_outcome.csv`: Simulated data with shuffled outcome
- `simulated_data_big_sample_reduced_mean5.csv`: Shifted dataset with Medd reduced mean by 5
- `simulated_data_big_sample_reduced_mean10.csv`: Shifted dataset with Medd reduced mean by 10
- `simulated_data_big_sample_reduced_mean20.csv`: Shifted dataset with Medd reduced mean by 20
- `simulated_data_big_sample_reduced_mean30csv`: Shifted dataset with Medd reduced mean by 30



## How to Run the Model File

To run the `training_models.py` file:
1. Change the path to your local machine on line 384: `path = <yourPath>`
2. Change the path to `experiment.yaml` on your local machine  on line 388.
3. Create a "results" folder under `<yourPath>` to store the test result. An example of the model result can be found here: [test_result_weighted.csv](https://www.dropbox.com/s/hrujn6zesf9ath9/test_result_weighted.csv?dl=0)

4. To use the weighted loss function model, Change the path to `experiment.yaml` on your local machine  on line 388. Rememer also to change the result file names.


