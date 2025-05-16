# FinalProject
This final project focuses on creating 4 different types of regression models based on meteorlogical data in the forestfires.csv data set. We will then be comparing those models based on four metrics mean RMSE (log), std RMSE (log), mean R2 (log), and std R2 (log). We will be using temp (Celsius), RH (percentage), wind (km/hr) speed, and rain (mm/m2) as the independent variables. We will also be using the dependent variable area (hectares). We will also be applying a logarithmic transformation on the area variable to stabilize the variance within the area variable. We will also be comparing the three metrics of each model before and after applying logartihimc transformation on the area variable. The 4 types of models that will be compared are: linear regression model, decision tree model, random forest model, and support vector machine model. 

# Files Included
'src/' (Source Code)

- config.py
- data_loader.py
- model_evaluation.py
- plotting.py
- utils.py

'statistic_tables/' (Contains .txt files containg the mean RMSE, std RMSE, and mean R2 for each of the 4 models)

- train_test_scores.txt
- tuned_log.txt
- tuned_raw.txt
- untuned_log.txt
- untune_raw.txt

'visualizations/'

- r2_mean_comparison.svg
- r2_std_comparison.svg
- rmse_comparison.svg
- rmse_std_comparison.svg
- target_distributions.svg

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

4. A new folder should pop up called 'results' containing all of the visualizations (.svg files) and the statistic tables (.txt files). 
