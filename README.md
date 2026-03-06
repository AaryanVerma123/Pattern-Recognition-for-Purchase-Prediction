# Pattern-Recognition-for-Purchase-Prediction
This project implements a simple machine learning based pattern recognition system. The objective is to predict whether a customer will purchase a product based on various features eg Age, Gender, Income Level, Product Category, Ad Interaction, Discount Offered, and Total Previous Purchases (TPP), 


## Dataset Requirements

The dataset must: - Be in CSV format - Contain input feature columns -
Contain a target column named `Purchase`

## Features Used

-   Age
-   Gender
-   Income_Level
-   Product_Category
-   Ad_Interaction
-   Discount_Offered in(%)
-   TPP in Past

### Stage 1 -- Training

1.  Load dataset
2.  Encode categorical features using LabelEncoder
3.  Split dataset into training and testing sets
4.  Train model (Logistic Regression / Random Forest)
5.  Evaluate model
6.  Save trained model using pickle

### Stage 2 -- Prediction

1.  Load saved model
2.  Accept new customer input
3.  Transform input using saved encoders
4.  Predict purchase outcome
5.  Display prediction result

## How to Run

### Train Model : - 
you can choose which model to train and use but for that you need to load the dataset into the model 

trainer model 1 Logistic Regg.py

df = pd.read_csv("enter your dataset here ")

trainer model 2 random forest.py

df = pd.read_csv("enter your dataset here")


### Model saving : -
after loading and running the model you will get a message in the terminal that your model is saved successfully

these are the name of your trained model which are saved in pkl format

1. logistic_model.pkl
2. random_forest_model.pkl


## loading the model into the predictor model : -
to load the model into the predictor you need to copy and past the names of the saved model which are 1. logistic_model.pkl 2. random_forest_model.pkl into the code

import pickle
import pandas as pd

with open("enter your model name here", "rb") as f:
    model, le_gender, le_income, le_product, le_ad, le_purchase = pickle.load(f)

## input and output
Once you run the code you will have to input the values in the terminal in this series

Age :- (type any age)

Gender:- (Male or female)

income level:- (High/Medium/Low)   ##(type from these 3 (please don't enter any amount))

enter product category:- (enter the name of any category from these ( Designer items,Groceries,Car,Electronics,Beauty Products))

Ad interaction :- (Yes or No)  #(type either yes or no according to your case)

Discount offered in %:- (only put the int value e.g.:- 1,2,3,4,5....)

TPP in past :-(enter any value 1,2,3,4....)  (this is total purchases in past which means if the customers is a regular or new)

Output :-
once you fill out all the inputs you will get the answer in either yes or no as shown below

Enter Age: 25
Enter Gender (Male/Female): male
Enter Income Level (High/Medium/Low): high
Enter Product Category: car
Ad Interaction (Yes/No): yes
Discount Offered (%): 10
TPP in Past: 5

Will customer purchase? Yes
Prediction Probability: [[0.385 0.615]]

## Author

Aaryan Verma
