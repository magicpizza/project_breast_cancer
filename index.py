import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import statsmodels.api as sm
import statsmodels.formula.api as smf

# setting graph styles
# %matplotlib inline
# rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# locate and read data
address='C:/Users/victo/Dropbox/Coding/Python/visual_studio_code/project_breast_cancer/breast_cancer_data.csv'
df=pd.read_csv(address)
# column_names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
# df.columns=column_names
# used https://github.com/patrickmlong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/blob/master/Breast%20Cancer%20Wisconsin%20(Diagnostic)%20DataSet_Orignal_Data_In_Progress.ipynb as a guide

# saving a string version of Diagnosis for future reference (when plotting)
diagnosis_str = df.diagnosis
# remapping M and B to 1 and 0 respectively
diagnosis_coder = {'M':1, 'B':0}
df.diagnosis = df.diagnosis.map(diagnosis_coder)
diagnosis = df.diagnosis

df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis = 1, inplace = True)
df['diagnosis'] = diagnosis
print(df.head())

# _______________________________________________
# Visualise data
# _______________________________________________
diagnosis_value_counts=df['diagnosis'].value_counts()
t_diagnosis=pd.DataFrame(diagnosis_value_counts)
t_diagnosis['percent']=100*diagnosis_value_counts/sum(diagnosis_value_counts)
print(t_diagnosis)
diagnosis_value_counts.plot(kind='bar')
# plt.show()
print('As shown, there are more benign than malignant cases')

# create list of df column names
mean_features = list(df.columns[0:10])

df_10 = df.loc[:, mean_features]
df_10['diagnosis_str']=diagnosis_str

print(df_10.head())

sb.pairplot(df_10, hue='diagnosis_str', palette='hls')
plt.show()

# _______________________________________________
## Logistic Regression
# _______________________________________________
# replace with for loop
radius_mean = df["radius_mean"]
texture_mean = df["texture_mean"]
perimenter_mean = df["perimeter_mean"]
area_mean = df["area_mean"]
smoothness_mean = df["smoothness_mean"]
compactness_mean = df["compactness_mean"]
concavity_mean = df["concavity_mean"]
concave_points_mean = df["concave points_mean"]
symmetry_mean = df["symmetry_mean"]
fractal_dimension_mean = df["fractal_dimension_mean"]

y=diagnosis
x=np.column_stack((radius_mean,texture_mean,perimenter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean))
x=sm.add_constant(x,prepend=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
logreg=LogisticRegression().fit(x_train,y_train)
logreg
print("Training set score: {:.3f}".format(logreg.score(x_train,y_train)))
print("Test set score: {:.3f}".format(logreg.score(x_test,y_test)))

logit_model=sm.Logit(y_train,x_train)
result=logit_model.fit()
print(result.summary())



# # create boolean DataFrame of Ben and Mal
# df_b = df[df['diagnosis'] == 0]
# df_m = df[df['diagnosis'] == 1]





## pairplot
# df_12=df.iloc[:,1:12]
# print(df_12.head())

## pairplot separating tumour nucleus characteristics based on diagnosis. Malignant in red
# plot is showing distinct patterns
# radius/area/perimeter/compactness/concavity/concave_points features highly predictive
# smoothness/symmetry are very homogenous
# concavity and concave_points seem to have the strongest positive relationship with other variables. 
# sb.pairplot(df_12, hue='diagnosis', palette='hls')
# # sb.pairplot(df_12)
# plt.show()


# # create a base classifier used to evaluate a subset of attributes
# model = LogisticRegression()
# # create the RFE model and select 3 attributes
# rfe = RFE(model, 3)
# rfe = rfe.fit(df_12.data, df_12.target)
# # summarize the selection of the attributes
# print(rfe.support_)
# print(rfe.ranking_)