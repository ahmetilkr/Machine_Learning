
################################################################
# İş Problemi:
################################################################
# Scoutlar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme.


################################################################
# Veriseti Hikayesi:
################################################################

# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların,
# maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.
# attributes: Oyuncuları değerlendiren kullanıcıların bir maçta izleyip değerlendirdikleri her oyuncunun özelliklerine
# verdikleri puanları içeriyor. (bağımsız değişkenler)
#potential_labels: Oyuncuları değerlendiren kullanıcıların her bir maçta oyuncularla ilgili nihai görüşlerini
# içeren potansiyel etiketlerini içeriyor. (hedef değişken)
# 9 Değişken, 10730 Gözlem, 0.65 mb


################################################################
# Değişkenler:
################################################################

# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi.

# match_id: İlgili maçın id'si.

# evaluator_id: Değerlendiricinin(scout'un) id'si.

# player_id: İlgili oyuncunun id'si.

# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id'si.

# 1- Kaleci
# 2- Stoper
# 3- Sağ bek
# 4- Sol bek
# 5- Defansif orta saha
# 6- Merkez orta saha
# 7- Sağ kanat
# 8- Sol kanat
# 9- Ofansif orta saha
# 10- Forvet


# analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme.

# attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si.

# attribute_value: Bir scoutun bir oyuncunun bir özelliğine verilen değer(puan).

# potential_label: Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)




import pandas as pd
import numpy as np
import matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


pd.set_option('display.max_columns', None)

################################################################
# Görev 1: Veri Setinin Hazırlanması
################################################################

################################################################
# Adım 1: 2 csv dosyasını okutunuz.
################################################################

df = pd.read_csv("machine_learning/datasets/scoutium_attributes.csv", sep=";")
df2 = pd.read_csv("machine_learning/datasets/scoutium_potential_labels.csv", sep=";")
df.head()
df2.head()

################################################################
# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
################################################################

dff=pd.merge(df,df2,how="left",on=["task_response_id", "match_id", "evaluator_id", "player_id"])
dff.head()


################################################################
# Adım 3: position_id içerisindeki Kaleci (1) sınıfını verisetinden kaldırınız.
################################################################

dff=dff.drop(index=dff[dff["position_id"]==1].index)
dff.head()


################################################################
# Adım 4: potential_label içerisindeki below_average sınıfını verisetinden kaldırınız.( below_average sınıfı
# tüm verisetinin %1'ini oluşturur)
################################################################


dff=dff.drop(index=dff[dff["potential_label"]=="below_average"].index)


################################################################
#Adım 5: Oluşturduğunuz verisetinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz.
# Bu pivot table'da her satır bir oyuncu olacak şekilde manipülasyon yapınız.
################################################################


pt=pd.pivot_table(data=dff,values="attribute_value",index=["player_id","position_id","potential_label"],columns="attribute_id")
pt.head()

################################################################
#“reset_index” fonksiyonunu kullanarak index hatasından kurtulunuz ve “attribute_id” sütunlarının isimlerini stringe çeviriniz. (df.columns.map(str))
################################################################

pt=pt.reset_index()
pt.columns=pt.columns.map(str)  #map fonksiyonu elindeki listeyi(pandas) içine verdiğin fonksiynu uygular


#Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

def label_encoder(dataframe, col):
    labelencoder = LabelEncoder()
    dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe


pt = label_encoder(pt,"potential_label")

################################################################
# Görev 3: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye kaydediniz.
################################################################


num_cols = pt.columns[3:]


################################################################
# Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için standardScaler uygulayınız.
################################################################

scaler=StandardScaler()

pt[num_cols]=scaler.fit_transform(pt[num_cols])
pt.head()

#Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
#geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)


pt.head()
x=pt.drop(["potential_label", "player_id"],axis=1)
y=pt["potential_label"]

models = [['LR', LogisticRegression()],
                   ['KNN', KNeighborsClassifier()],
                   ["RF", RandomForestClassifier()],
                   ['GBM', GradientBoostingClassifier()],
                   ['XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')],
                    ["LightGBM", LGBMClassifier()]]



for name, model in models:
    print(name)
    cv_results=cross_validate(model,
                   X=x,
                   y=y,
                   scoring=["roc_auc", "f1", "precision", "recall", "accuracy"],
                   cv=10)

    print(f"ROC AUC  : {cv_results['test_roc_auc'].mean():.4f}")
    print(f"F1       : {cv_results['test_f1'].mean():.4f}")
    print(f"Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"Recall   : {cv_results['test_recall'].mean():.4f}")
    print(f"Accuracy : {cv_results['test_accuracy'].mean():.4f}\n")





lgbm_model = LGBMClassifier(random_state=46)


lgbm_model.fit(x,y)



results=cross_validate(lgbm_model,x,y,cv=5,scoring=["roc_auc", "f1", "precision", "recall", "accuracy"])
print(f"ROC AUC  : {results['test_roc_auc'].mean():.4f}")
print(f"F1       : {results['test_f1'].mean():.4f}")
print(f"Precision: {results['test_precision'].mean():.4f}")
print(f"Recall   : {results['test_recall'].mean():.4f}")
print(f"Accuracy : {results['test_accuracy'].mean():.4f}\n")


################################################################
# Görev 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################

# feature importance
def plot_importance(model, features, num=len(x), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(lgbm_model, x)




































