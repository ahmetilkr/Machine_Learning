import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
matplotlib.use('Qt5Agg')

df_ = pd.DataFrame(
    {
        "Gerçek Değer": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "Model Olasılık Tahmini": [
            0.7,
            0.8,
            0.65,
            0.9,
            0.45,
            0.5,
            0.55,
            0.35,
            0.4,
            0.25,
        ],
    })

df_["model_tahmin"]=np.where(df_["Model Olasılık Tahmini"]>=0.5,1,0)

#######################################
#GÖREV1
#######################################
ConfusionMatrixDisplay.from_predictions(df_["Gerçek Değer"],df_["model_tahmin"] , cmap="Blues")



TP=5
TN=3
FP=1
FN=1
TOPLAM=10
accuracy=(TP+TN)/TOPLAM
recall=TP/(TP+FN)
precision=TP/(TP+FP)
f1=(2*(precision*recall))/(precision+recall)

print(classification_report(df_["Gerçek Değer"],df_["model_tahmin"]))


#######################################
#GÖREV2
#######################################
TP_2 = 5
FN_2 = 5
FP_2 = 90
TN_2 = 900
TOPLAM_2=1000
accuracy_2=(TP+TN)/TOPLAM
recall_2=TP/(TP+FN)
precision_2=TP/(TP+FP)
f1_2=(2*(precision*recall))/(precision+recall)



# Ekip sadece %90.5 lik yüksek doğruluk oranına aldanarak asıl tabloyu gözden kaçırmıştır.
# çünkü 1000 işlemin zaten 990 ı normal olduğu için model neredeyse hiçbir şey yapmadan da yüksek skor alabilmektedir.
# Banka için en önemli şey dolandırıcıları yakalamakken, model gerçekteki 10 dolandırıcılığın 5'ini elinden kaçırmış,
# dolandırıcı dediği 95 masum işlemin 90'ında yanılmıştır.Yani sadece accuracy değerine bakarak precision,recall ve f1 metriklerini göz ardı ederek yanılmıştır


