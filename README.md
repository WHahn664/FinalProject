# FinalProject
This final project focuses on creating 4 different types of regression models based on meteorlogical data in the forestfires.csv data set. We will then be comparing those models based on four metrics mean RMSE (log), std RMSE (log), mean R2 (log), and std R2 (log). We will be using temp (Celsius), RH (percentage), wind (km/hr) speed, and rain (mm/m2) as the independent variables. We will also be using the dependent variable area (hectares). We will also be applying a logarithmic transformation on the area variable to stabilize the variance within the area variable. We will also be comparing the four metrics of each model before and after applying logartihimc transformation on the area variable. We will also be hypetuning our models (exclusing linear regression) by performing a GridSearchCV to find the most optimal hyperparameters to use for our models. The 4 types of models that will be compared are: linear regression model, decision tree model, random forest model, and support vector machine model. Check the pdf file in the writeup file for more details about the project and conclusions. 

# Files Included

'writeup/'

- forestfires.csv: Contains the data set used for this project.

'src/' (Source Code)

- config.py: This contains the range of hyperparameters that we will be using for our models. We will then use GridSearchCV in the model_evaluation.py file to help us choose the most optimal hyperparameters for our models (excluding linear regression).
- data_loader.py: This contains code that loads the forestfires.csv data set. It also contains code that splits the into a training set (80% of the data) and a testing set (20% of the data).
- model_evaluation.py: This contains code that will help generate the statistical tables.
- plotting.py: This contains code that will help us with plotting our bar plots and saving them as .svg files in a new folder called "results".
- utils.py: This contains code that will help us save our statistical .txt files into a new folder called "results". 

'statistic_tables/' (Contains .txt files containing the mean RMSE, std RMSE, and mean R2 for each of the 4 models)

- train_test_scores.txt: Contains summary tables of each model's evaluation metrics using a single train/test split. It also contains the results of how each model performed before and after hypertuning. It also contains the results of how each model performed for the raw area variable and the log-transformed area variable.
- tuned_log.txt: Contains a summary table of the each model's statistical metrics using a 5-fold cross validation with hypertuning. Each model uses the log-transformed area variable.
- tuned_raw.txt: Contains a summary table of the each model's statistical metrics using a 5-fold cross validation with hypertuning. Each model uses the raw area variable. 
- untuned_log.txt: Contains a summary table of the each model's statistical metrics using a 5-fold cross validation without hypertuning. Each model uses the log-transformed area variable. 
- untuned_raw.txt: Contains a summary table of the each model's statistical metrics using a 5-fold cross validation without hypertuning. Each model uses the raw area variable. 

'visualizations/'(Contains the .svg files containing the bar plots of the 4 statistical metrics. It also contains a bar plot comparing the raw area distributions versus the log-transformed area distributions.)

- r2_mean_comparison.svg: Contains a bar plot representation of each model's mean R2 scores using a 5-fold cross validation with and without hypertuning. It also compares those results for the raw area variable and log-transformed area variable. 
- r2_std_comparison.svg: Contains a bar plot representation of each model's std R2 scores using a 5-fold cross validation with and without hypertuning. It also compares those results for the raw area variable and log-transformed area variable. 
- rmse_comparison.svg: Contains a bar plot representation of each model's mean RSME scores using a 5-fold cross validation with and without hypertuning. It also compares those results for the raw area variable and log-transformed area variable. 
- rmse_std_comparison.svg: Contains a bar plot representation of each model's std RSME scores using a 5-fold cross validation with and without hypertuning. It also compares those results for the raw area variable and log-transformed area variable. 
- target_distributions.svg: Contains a bar plot comparing the raw area distributions to the the log-transformed area distributions.

'writeup/'

- Predicting Wildfires.pdf: Contains a detailed writeup about the project and results.

main.py: Contains the starting code that initiates the analysis. Also puts your .txt tables and .svg files in a separate folder called "results."

# Steps to evaluate the models:

1. Clone github repository in the terminal on jupyterhub (or any other editor) like this:
git clone https://github.com/WHahn664/FinalProject.git
2. Run the main.py in the terminal (this might take a few minutes to run completely):
python main.py

3 (Side note to what I did in jupyterhub). Or you can do what I did and run something like this in the terminal):
python /home/jovyan/FinalProject/main.py

4. A new folder should pop up called 'results' containing all of the visualizations (.svg files) and the statistic tables (.txt files). 

* As an additional note, check the pdf file in the writeup file just in case to see if it clones correctly.
