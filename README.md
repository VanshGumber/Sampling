# Sampling Techniques vs ML Model Performance

## Objective
This assignment studies how different sampling techniques affect the performance of machine learning models when working with an imbalanced dataset.

## Dataset
Credit card fraud dataset provided in the assignment:  
https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

## Steps Performed
1. Loaded the imbalanced dataset  
2. Balanced the dataset using SMOTE  
3. Split the data into training and testing sets  
4. Created five samples using different sampling techniques:
   - Simple Random Sampling  
   - Stratified Sampling  
   - Systematic Sampling  
   - Cluster Sampling  
   - Bootstrap Sampling  
5. Trained five machine learning models:
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Naive Bayes  
   - Support Vector Machine  
6. Measured model accuracy for each sampling technique  
7. Visualized results using heatmap and bar chart  

## Output
The program produces:
- Accuracy comparison table
- Heatmap of model performance
- Bar chart comparing sampling techniques

## Conclusion
Different sampling techniques affect models differently.  
The best sampling method depends on the model used and the data distribution.

## How to Run
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn  
python sampling.py
