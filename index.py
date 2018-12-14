import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale

# locate and read data
address='C:/Users/victo/Dropbox/Coding/Python/visual_studio_code/project_breast_cancer/breast_cancer_data.csv'
df=pd.read_csv(address)
# column_names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
# df.columns=column_names
# used https://github.com/patrickmlong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/blob/master/Breast%20Cancer%20Wisconsin%20(Diagnostic)%20DataSet_Orignal_Data_In_Progress.ipynb as a guide








