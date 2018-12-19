# _______________________________________________
## Introduction
# _______________________________________________
# Creating a prognostic model from Wisconsin Breast Cancer Data 
# by Victor Wan
# Desc: Visualising Breast Cancer Wisconsin data and creating a predictive model based on nuclear features

# Importing libraries

print('Creating a prognostic model from Wisconsin Breast Cancer Data\n~by Victor Wan\nDesc: Visualising Breast Cancer Wisconsin data and creating a predictive model based on nuclear features')
# used to find breast_cancer_data.csv
import os

# numpy is used to manipulate arrays (used in this project for .column_stack())
import numpy as np
# panda for data analysis (used for reading in data and converting to DataFrame)
import pandas as pd

# libraries for plotting data
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

# libraries for logistic regression
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Setting graph styles
# %matplotlib inline
# rcParams['figure.figsize'] = 5, 4
print('Setting graph styles...')
sb.set_style('whitegrid')

# Locate and read data
print('Locating and reading data...')
address = os.path.realpath(
    os.path.join(os.getcwd(), 'breast_cancer_data.csv'))
df=pd.read_csv(address)
# column_names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
# df.columns=column_names
# used https://github.com/patrickmlong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/blob/master/Breast%20Cancer%20Wisconsin%20(Diagnostic)%20DataSet_Orignal_Data_In_Progress.ipynb as a guide

# define functions
print('Defining functions...')
def avg(a):
    '''returns an average of counts stored in a series
    '''
    return(a/sum(a))

def count_categorical_to_df(a):
     '''input a categorical variable and create a DataFrame of counts for each level of the categorical variable. 
     '''
     value_counts=a.value_counts()
     return(pd.DataFrame(value_counts))

def append_first_10_columns_to_tuple(a):
     '''input a dataframe and create a 2D tuple with only columns 0-9
     '''
     columns_1_to_10_list = []
     for column in range(10):
          columns_1_to_10_list.append(a.iloc[:,column])
     return(tuple(columns_1_to_10_list))

def non_numeric_to_NA(col):
     '''Check whether a column's values are numeric, and if not numeric, modify to NA. 
     ''' 
     for id in range(568):
        if type(col[id]) == int or type(col[id]) == np.float64:
            pass
        else:
            col[id]=np.float64(col[id])

## creating string and binary diagnosis variables
# saving string version of Diagnosis for future reference (when plotting)
diagnosis_str = df.diagnosis
# remapping M and B to 1 and 0 respectively
diagnosis_coder = {'M':1, 'B':0}
df.diagnosis = df.diagnosis.map(diagnosis_coder)
diagnosis_int = df.diagnosis
# create separate dataframes for graphing later on
df_b = df[df['diagnosis'] == 0]
df_m = df[df['diagnosis'] == 1]

# dropping unnecessary columns
# ID is not necessary for analysis, diagnosis is removed for rearranging, Unnamed: 32 is an unknown column. 
df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis = 1, inplace = True)
df['diagnosis'] = diagnosis_int
# checking if all values in df.texture_mean are numpy.float64, and converting to NA if false 
non_numeric_to_NA(df.texture_mean)

# peeking at data
print('Peeking at data...')
print(df.head())
print(df.info())

# _______________________________________________
## Visualise data
# _______________________________________________
print('Visualising data...')
print('Visualising the proportion of benign and malignant cases...')
# creating a dataframe table and bar chart comparing the amount of benign and malignant cases
t_diagnosis = count_categorical_to_df(df['diagnosis'])
diagnosis_value_counts=df['diagnosis'].value_counts()
t_diagnosis=pd.DataFrame(diagnosis_value_counts)
t_diagnosis['percent']=100*avg(diagnosis_value_counts)
print(t_diagnosis)
diagnosis_value_counts.plot(kind='bar')
print('There are more benign than malignant cases in the Wisconsin dataset')

# Create list of df column names
mean_features = []
for column in df.columns[0:10]:
     mean_features.append(column)

# Create dataframe where only mean features and diagnosis are included
df_10 = df.loc[:,mean_features]
df_10['diagnosis_str']=diagnosis_str

# creating a pairplot of data
print('Creating pairplot of data...')
sb.pairplot(df_10, hue='diagnosis_str', palette='hls')
plt.show()

# Creating a matrix of boxplots for mean features
print('Creating histograms showing distribution when separated by benign and malignant cases...')
fig = plt.figure()
for i,b in enumerate(list(df.columns[0:10])):
    # enumerate starts at index 0, need to add 1 for subplotting
    i +=1
    # creating subplots
    ax = fig.add_subplot(3,4,i)
    ax.boxplot([df_b[b], df_m[b]])
    ax.set_title(b)
plt.tight_layout()
plt.legend()
plt.show()
print('Plots show distinct patterns\n1. radius/area/perimeter/compactness/concavity/concave_points features have distinct Bemign and Malignant populations\n2. Smoothness/symmetry are very homogenous\nConcavity and concave_points seem to have the strongest positive relationship with other variables.')

# _______________________________________________
## Logistic Regression
# _______________________________________________
print('Performing logistic regression analysis...')
# creating a tuple dataframe for the first 10 columns of df (ie. the columns which show mean characteristics). 
columns_1_to_10_tuple = append_first_10_columns_to_tuple(df)

# defining the x and y variables for logistic regression
y = diagnosis_int
x = np.column_stack(columns_1_to_10_tuple)
x = sm.add_constant(x,prepend=True)

# creating logistic regression
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
logreg = LogisticRegression().fit(x_train,y_train)
logreg
print("Training set score: {:.3f}".format(logreg.score(x_train,y_train)))
print("Test set score: {:.3f}".format(logreg.score(x_test,y_test)))

# create a confusion matrix
y_predict = logreg.predict(x_test)
print(classification_report(y_test,y_predict))

# cross validating
scores = cross_val_score(logreg, x_train, y_train, cv=5)
scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('\nConducted logistic regression because the output only has two possibilities. The model does not assume normal distrubution, which is ideal as the pair plot shows some skewed distributions.\nA random forest would also have worked but a logistic regression is faster and more interpretable. This is significant considering the size of the dataframe.\nAlso the accuracy is high whilst the sd of the accuracy is small') 