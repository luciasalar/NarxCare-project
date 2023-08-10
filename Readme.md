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

Important Note: 
Please make sure to adjust the feature names in the `experiment.yaml` file or column names of the input file accordingly. This is important to ensure that the `experiment.yaml` file selects the correct features for training the model. 

In particular, the feature set 1 should include variables as specified in table 1 of the manuscript, excluding variables that are relevant to opioid dependency but not in NarxCare ORS (also documented in table 1).

The feature set 2 should include all variables listed in table 1 of the manuscript.

#### Model Files
- `training_models.py`:  training model

## Simulated Data (Folder, in a separate dropbox link):
We simulated a population sample consisting of 2 million observations, 52 independent variables, and one dependent (outcome) variable (Opioid Overdose incident in the past year). 

The simulated dataset contains the following variables:

- Ux_OP_PoC_ED: The slope of emergency Department stops.
- Ux_OP_PoC_UC: The slope of urgent care stops.
- Ux_OP_PoC_TobaccoCessation: The slope of tobacco cessation stops.
- Ux_TotalNumOfAppt: The slope of total appointments.
- Ux_OP_MHOC_HBPC: The slope of home based primary care.
- Rx_OpioidForPain_0to182_D: Opioids For Pains in the past 0 to 182 days.
- Rx_OpioidForPain_182to365_D: Opioids For Pains in the past 182 to 365 days.
- Rx_Insomnia_0to365_D:  Whether the patient was on an Insomnia drug in the past 365 days.
- Paton_Rx_Insomnia: Whether the patient was on an Insomnia drug at the time of selection.
- Dx_SAE_Acetaminophen_0to1_Y: Whether the patient had a Serious Adverse Event - Acetaminophen in the past one year.
- Dx_SAE_Falls_0to1_Y: Whether the patient had a Serious Adverse Event - fall in the past one year.
- Dx_SAE_OtherAccident_0to1_Y: Whether the patient had a Serious Adverse Event - Other Accidents in the past one year.
- Dx_SAE_OtherDrug_0to1_Y: Whether the patient had a Serious Adverse Event - Poisoning by Other drugs in the past one year.
- Dx_SAE_Sedative_0to1_Y: Whether the patient had a Serious Adverse Event - Poisoning by Sedative in the past one year.
- Dx_SAE_Vehicle_0to1_Y: Whether the patient had a Serious Adverse Event - Vehicular Accidents in the past one year.
- Dx_SedativeUD_0to1_Y: Whether the patient had Sedative Use Disorder in the past one year.
- Dx_SUD_CatchAll_RV2_0to1_Y: Whether the patient had Substance use disorders not included in other definitions in the dataset.
- Dx_AlcoholUD_0to1_Y: Whether the patient had Alcohol Use Disorder in the past 1 year.
- Dx_Amphetamine_OtherStim_UD_0to1: Whether the patient had Amphetamine and other Stimulant Disorder.
- Rx_Medd: Morphine equivalent daily dose on index date.
- Rx_Medd_10m: MME Morphine equivalent daily dose 10 months ago.
- Rx_Medd_11m: MME Morphine equivalent daily dose 11 months ago.
- Rx_Medd_1m: MME Morphine equivalent daily dose 1 months ago.
- Rx_Medd_2m: MME Morphine equivalent daily dose 2 months ago.
- Rx_Medd_3m: MME Morphine equivalent daily dose 3 months ago.
- Rx_Medd_4m: MME Morphine equivalent daily dose 4 months ago.
- Rx_Medd_5m: MME Morphine equivalent daily dose 5 months ago.
- Rx_Medd_6m: MME Morphine equivalent daily dose 6 months ago.
- Rx_Medd_7m: MME Morphine equivalent daily dose 7 months ago.
- Rx_Medd_8m: MME Morphine equivalent daily dose 8 months ago.
- Rx_Medd_9m: MME Morphine equivalent daily dose 9 months ago.
- Rx_Anxiolytic_0to365_D: Whether the patient was on an Anxiolytic drug in the past one year.
- Rx_OAT_methadone_12m: Whether the patient had a methadone prescription in the past 12 months.
- Rx_OAT_buprenorphine_12m: Whether the patient had a buprenorphine prescription in the past 12 months.
- agegroup_ls30: Whether the patient's age is less than 30.
- agegroup_ge30: Whether the patient's age is between 30 and 40.
- agegroup_ge40: Whether the patient's age is between 40 and 50.
- agegroup_ge50: Whether the patient's age is between 50 and 60.
- agegroup_ge60: Whether the patient's age is between 60 and 70.
- agegroup_ge70: Whether the patient's age is above 70.
- sex_female: Whether the patient is female.
- race_White: Whether the patient is of White race.
- Dx_UDS1_12m: Whether the patient had a Urine drug screen for heroin/morphine in the past one year.
- Dx_UDS2_12m: Whether the patient had a Urine drug screen for nonmorphine opioid compounds in the past one year.
- Dx_UDS3_12m: Whether the patient had a Urine drug screen for nonopioid abusable substances in the past one year.
- MH_Psychoses: Whether the patient had Psychoses in the past one year.
- MH_MDD_12m: Whether the patient had MDD (Major Depressive Disorder) in the past one year.
- MH_Psychoses: Whether the patient had Psychoses in the past one year.
- MH_PTSD_1to2_Y: Whether the patient had PTSD (Post-Traumatic Stress Disorder) in the past one year.
- PA_Pain_Urogenital_0to1_Y: Whether the patient had Urogenital pain in the past one year.
- PA_Pain_Neuropathy_0to1_Y: Whether the patient had Neuropathy pain in the past one year.
- PA_Pain_Back: Whether the patient had Back pain in the past one year.
- PA_Pain_Neck_0to1_Y: Whether the patient had Neck pain in the past one year.
- PA_Pain_Fibromyalgia_0to1_Y: Whether the patient had Fibromyalgia pain in the past one year.
- PA_Endometriosis_0to1_Y: Whether the patient had Endometriosis in the past one year.
- EH_Homeless_1to2_Y: Whether the patient experienced homelessness in the past 1 to 2 year.
- EH_RehMedPain: Whether the patient received care at a pain clinic in the past 12 months.
- EH_Obesity_1to2_Y: Whether the patient is obese in the past 1 to 2 year.
- Dx_OpioidOverdose_0to1_Y (Outcome): Whether the patient had an Opioid Overdose in the past one year.


### Datasets
Creation process of the datasets are describe in the manuscript section 4.2 

- `simulated_data_big_sample.csv`: Simulated data
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


# Experiments
This repository contains a paper and its accompanying experiments. The experiments conducted are as follows:

### Experiments Conducted
1. Run both the weighted and non-weighted surrogate algorithm in the `training_models.py` file on the simulated dataset (`simulated_data_big_sample.csv`). This task has been completed and the results are documented on manuscript section 5.1 - 5.4.

2. Run both the weighted and non-weighted surrogate algorithm in the `training_models.py` file on the simulated dataset with shifted variables, including `simulated_data_big_sample_reduced_mean5.csv`, `simulated_data_big_sample_reduced_mean10.csv`, `simulated_data_big_sample_reduced_mean20.csv`, and `simulated_data_big_sample_reduced_mean30.csv`. This task has been completed, and the results are documented on manuscript section 5.5.

3. Conduct a falsification test on simulated data with shuffled outcome using the `simulated_data_big_sample_shuffled_outcome.csv` file. This task has been completed, and the results are documented on manuscript section 5.6.

### Remaining Tasks

#### Run both the weighted and non-weighted surrogate algorithm in the `training_models.py` file on the VA dataset.

1.  Recode the VA variable name or adjust the variable selection in the `experiment.yaml` file to ensure that the algorithm correctly selects feature set 1 and set 2. The feature sets are documented in the manuscript.

2.  Modify the path configuration in the `training_models.py` file to match your local machine, and ensure that the algorithm correctly selects feature set 1 and set 2. The feature sets are documented in the manuscript.

3. Modify the path configuration in the `training_models.py` file to match your local machine, and complete the remaining tasks as necessary.

