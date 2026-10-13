import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
matplotlib.use('Qt5Agg')

df=pd.DataFrame({"Deneyim_Yılı":[5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
              "Maaş":[600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]})

df.head()

Bias=275
Weight=90

X=df[["Deneyim_Yılı"]]
y=df["Maaş"]


y_pred=Weight*df["Deneyim_Yılı"]+Bias

mean_squared_error(y,y_pred)
root_mean_squared_error(y,y_pred)
mean_absolute_error(y,y_pred)


(np.square(y-y_pred).sum())/len(y)
np.sqrt((np.square(y-y_pred).sum())/len(y))
(np.abs(y-y_pred).sum())/len(y)


