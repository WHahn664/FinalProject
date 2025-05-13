# FinalProject
This final project focuses on creating 4 different types of regression models based on meteorlogical data in the forestfires.csv data set. We will then be comparing those models based on three metrics mean RMSE (log), std RMSE (log), mean R2 (log). We will be using temp (Celsius), RH (percentage), wind (km/hr) speed, and rain (mm/m2) as the independent variables. We will also be using the dependent variable area (hectares). We will also be applying a logarithmic transformation on the area variable to stabilize the variance within the area variable. We will also be comparing the three metrics of each model before and after applying logartihimc transformation on the area variable. The 4 types of models that will be compared are: linear regression model, decision tree model, random forest model, and support vector machine model. 

# Files Included
'src/' (Source Code)

- config.py
- data_loader.py
- model_evaluation.py
- plotting.py
- utils.py

'statistic_tables/' (Contains .txt files containg the mean RMSE, std RMSE, and mean R2 for each of the 4 models)

- train_test_scores_log.txt
- train_test_scores_raw.txt
- tuned_log.txt
- tuned_raw.txt
- untuned_log.txt
- untune_raw.txt

'visualizations/'

- rmse_comparison.svg

'writeup/'

- Final Project Draft by William Hahn.pdf


forestfires.csv

Predicting Forest Fires.ipynb

# Steps to evaluate the models:

1. Clone github repository in the terminal on jupyterhub (or any other editor) like this:
git clone https://github.com/WHahn664/FinalProject.git
2. Run the main.py in the terminal (this might take a few minutes to run completely):
python main.py

3 (Side note to what I did in jupyterhub). Or you can do what I did and run something like this in the terminal):
python /home/jovyan/great/FinalProject/main.py

4. A new folder should pop up contaiing the 4 csv containg the three metrics (mean RMSE, std RMSE, mean R2) for the 4 models for both the raw data (untransformed area column) and the logarthimic transformed data (log transformed area column). This folder also contains a single .svg file containing 4 bar blots showing the mean RMSE for the 4 models per bar plot. Two of the bar plots contain the mean RMSE for untuned models for the raw data. The other two bar plots contain the mean RMSE for tuned models for the logarthimic transformed data.
