'''
https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
https://machinelearningmastery.com/feature-selection-with-categorical-data/
https://machinelearningmastery.com/chi-squared-test-for-machine-learning/


There are two main types of feature selection techniques: supervised and unsupervised, 
        and supervised methods may be divided into wrapper, filter and intrinsic.
        Filter-based feature selection methods use statistical measures to score the correlation 
        or dependence between input variables that can be filtered to choose the most relevant features.

Feature selection methods are intended to reduce the number of input variables to those 
that are believed to be most useful to a model in order to predict the target variable.

1. One way to think about feature selection methods are in terms of supervised and unsupervised methods.

    1). Unsupervised feature selection techniques ignores the target variable, 
such as methods that remove redundant variables using correlation. 

    2). Supervised feature selection techniques use the target variable, such as methods 
that remove irrelevant variables.
    2.1) Wrapper feature selection methods create many models with different subsets of input features
        and select those features that result in the best performing model according to 
        a performance metric.
   2.2) Filter feature selection methods use statistical techniques to evaluate the relationship 
       between each input variable and the target variable, and these scores are used as the basis 
       to choose (filter) those input variables that will be used in the model.
   2.3) Finally, there are some machine learning algorithms that perform feature selection automatically
       as part of learning the model. We might refer to these techniques as intrinsic (embedded) feature selection methods.
       This includes algorithms such as penalized regression models like Lasso and decision trees, 
       including ensembles of decision trees like random forest.

   3). Feature selection is also related to dimensionally reduction techniques
          The difference is that feature selection select features to keep or remove from the dataset, 
          whereas dimensionality reduction create a projection of the data resulting in entirely 
          new input features. As such, dimensionality reduction is an alternate to 
          feature selection rather than a type of feature selection.


2. Statistics for Filter-Based Feature Selection Methods
    Common input variable data types:
        . Numerical Variables
        . Categorical Variables.
            . Boolean Variables (dichotomous).
            . Ordinal Variables.
            . Nominal Variables.

Numerical Output: Regression predictive modeling problem.
    Numerical Input, Numerical Output:
        Pearson’s correlation coefficient (linear).
        Spearman’s rank coefficient (nonlinear)
    Categorical Input, Numerical Output:
        ANOVA correlation coefficient (linear).
        Kendall’s rank coefficient (nonlinear).    
    
Categorical Output: Classification predictive modeling problem.
    Numerical Input, Categorical Output:
        ANOVA correlation coefficient (linear).
        Kendall’s rank coefficient (nonlinear).
    Categorical Input, Categorical Output:
        Chi-Squared test (contingency tables).
        Mutual Information.


3. Tips and Tricks for Feature Selection

    Correlation Statistics:
        The scikit-learn library provides an implementation of most of the useful statistical measures.

      For example:
        Pearson’s Correlation Coefficient: f_regression()
        ANOVA: f_classif()
        Chi-Squared: chi2()
        Mutual Information: mutual_info_classif() and mutual_info_regression()
      Also, the SciPy library provides an implementation of many more statistics, 
        such as Kendall’s tau (kendalltau) and Spearman’s rank correlation (spearmanr).

    Selection Method:
        Two of the more popular methods include:
            Select the top k variables: SelectKBest
            Select the top percentile variables: SelectPercentile


4. Worked Examples of Feature Selection

4.1 Regression Feature Selection: (Numerical Input, Numerical Output)
    Feature selection is performed using Pearson’s Correlation Coefficient via the f_regression() function
'''

#https://www.upgrad.com/blog/types-of-supervised-learning/

# pearson's correlation feature selection for numeric input and numeric output
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# generate dataset
X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
# define feature selection
fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)


'''
4.2 Classification Feature Selection: (Numerical Input, Categorical Output)
    Feature selection is performed using ANOVA F measure via the f_classif() function.
'''

# ANOVA feature selection for numeric input and categorical output
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# generate dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
# define feature selection
fs = SelectKBest(score_func=f_classif, k=2)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)


'''
4.3 Classification Feature Selection: (Categorical Input, Categorical Output)
The two most commonly used feature selection methods:
    chi-squared statistic
    mutual information statistic.
'''
# 4.3.1 Breast Cancer Categorical Dataset
filename = "breast-cancer.csv"
# example of loading and preparing the breast cancer dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
 
# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
	X = X.astype(str)
	return X, y
 
# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
 
# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)


# 4.3.2 Categorical Feature Selection
# 4.3.2.1 Chi-Squared Feature Selection

# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot

def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

'''
We could set k=4 When configuring the SelectKBest to select these top four features.
'''

# 4.3.2.2 Mutual Information Feature Selection

# example of mutual information feature selection for categorical data
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
 
# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
	X = X.astype(str)
	return X, y
 
# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
 
# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 
# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()


# 4.3.3 Modeling With Selected Features
'''Logistic regression is a good model for testing feature selection methods as it can 
   perform better if irrelevant features are removed from the model.'''
# 4.3.3.1 Model Built Using All Features
   
# evaluation of a model using all input features
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
	X = X.astype(str)
	return X, y
 
# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
 
# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# fit the model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_enc, y_train_enc)
# evaluate the model
yhat = model.predict(X_test_enc)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))   
   
# 4.3.3.2 Model Built Using Chi-Squared Features

# evaluation of a model fit using chi squared input features
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
	X = X.astype(str)
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k=4)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs

# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# fit the model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_fs, y_train_enc)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))


# 4.3.3.3 Model Built Using Mutual Information Features

# evaluation of a model fit using mutual information input features
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
	X = X.astype(str)
	return X, y
 
# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
 
# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k=4)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs
 
# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# fit the model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_fs, y_train_enc)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %.2f' % (accuracy*100))




'''
### Chi-square Original: ###
'''

import pandas as pd
import numpy as np 
import matplotlib as pl

data = {'sex': ['m','m','f','m','f','m','m','f','m','f'], \
        'grade':['a','a','c','b','a', 'a','a','c','b','a'],\
        'sub':['math','art', 'art','math','math','art', 'art','math','math','art',]}

df = pd.DataFrame(data)
df
df.describe() 
df.info() 
df.dtypes

# Contingency Table showing correlation between sex and grade
data_crosstab = pd.crosstab(df['grade'], 
                            df['sex'],  
                               margins = False) 
print(data_crosstab)

# Contingency Table showing correlation between Grades+Sub and sex.
data_crosstab = pd.crosstab([df['grade'], df['sub']],\
                             df['sex'], margins = False) 
print(data_crosstab)












