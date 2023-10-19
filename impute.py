from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

data = pd.read_csv('duke_gpa.csv').drop('gender', axis=1)


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(data)
data_imputed = imputer.transform(data)
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

print(data_imputed)
