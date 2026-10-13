##############################
# Telco Customer Churn Feature Engineering
##############################

# Problem : Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

# Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir.
# Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.

# 21 Değişken 7043 Gözlem

# CustomerId : Müşteri İd’si
# Gender : Cinsiyet
# SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
# Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
# Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
# tenure : Müşterinin şirkette kaldığı ay sayısı
# PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
# StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
# Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges : Müşteriden tahsil edilen toplam tutar
# Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler


# Her satır benzersiz bir müşteriyi temsil etmekte.
# Değişkenler müşteri hizmetleri, hesap ve demografik veriler hakkında bilgiler içerir.
# Müşterilerin kaydolduğu hizmetler - phone, multiple lines, internet, online security, online backup, device protection,
# tech support, and streaming TV and movies
# Müşteri hesap bilgileri – ne kadar süredir müşteri oldukları, sözleşme, ödeme yöntemi, kağıtsız faturalandırma, aylık ücretler
# ve toplam ücretler
# Müşteriler hakkında demografik bilgiler - cinsiyet, yaş aralığı ve ortakları ve bakmakla yükümlü oldukları kişiler olup olmadığı


# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
           # Adım 1: Genel resmi inceleyiniz.
           # Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
           # Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
           # Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
           # Adım 5: Aykırı gözlem analizi yapınız.
           # Adım 6: Eksik gözlem analizi yapınız.
           # Adım 7: Korelasyon analizi yapınız.

# GÖREV 2: FEATURE ENGINEERING
           # Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız.
           # işlemleri uygulayabilirsiniz.
           # Adım 2: Yeni değişkenler oluşturunuz.
           # Adım 3:  Encoding işlemlerini gerçekleştiriniz.
           # Adım 4: Numerik değişkenler için standartlaştırma yapınız.
           # Adım 5: Model oluşturunuz.


# Gerekli Kütüphane ve Fonksiyonlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import get_dummies
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import matplotlib


matplotlib.use("Qt5Agg")

import warnings


warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"]=np.where(df["Churn"] == "Yes",1,0)


def grab_col_names(dataframe,car_th=20,cat_th=10):
    num_cols=[col for col in dataframe.columns if str(dataframe[col].dtype) in ["int64","float64"]]
    cat_cols=[col for col in dataframe.columns if str(dataframe[col].dtype) in ["object","str","bool"]]
    num_but_cat =[col for col in dataframe.columns if str(dataframe[col].dtype) in ["int64","float64"] and dataframe[col].nunique() < cat_th]
    cat_but_car=[col for col in dataframe.columns if str(dataframe[col].dtype) in ["object","str"] and dataframe[col].nunique() > car_th]

    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]

    num_cols=[col for col in num_cols if col not in num_but_cat]

    return num_cols,cat_cols,num_but_cat,cat_but_car

num_cols,cat_cols,num_but_cat,cat_but_car=grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                  "radio":dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df,col,plot=True)


def num_summary(dataframe,col_name,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(8,5))
        dataframe[col_name].hist(bins=20)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)


for col in num_cols:
    num_summary(df,col,plot=True)

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

def target_summary_with_num(dataframe, target, col):
    print(dataframe.groupby(target).agg({col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


def target_summary_with_cat(dataframe, target, col):
    print(pd.DataFrame({"Target_Mean":dataframe.groupby(col)[target].mean(),
                  "Count":dataframe[col].value_counts(),
                  "Ratio":100 * dataframe[col].value_counts()/len(dataframe)}))

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


##################################
# AYKIRI DEĞER ANALİZİ
##################################

#Adım 5: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1=dataframe[col_name].quantile(q1)
    quartile3=dataframe[col_name].quantile(q3)
    IQR=quartile3-quartile1
    low_limit = quartile1 - 1.5 * IQR
    up_limit=quartile3 + 1.5 * IQR

    return low_limit,up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if ((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)).any():
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    df.loc[dataframe[variable] > up_limit, variable] = up_limit
    df.loc[dataframe[variable] < low_limit, variable] = low_limit


for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


##################################
# EKSİK DEĞER ANALİZİ
##################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_cols=[col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss=dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio=(dataframe[na_cols].isnull().sum()/len(dataframe)*100).sort_values(ascending=False)
    pd.concat([n_miss,ratio],axis=1,keys=["n_miss","ratio"])

    if na_name:
        return na_cols

na_columns = missing_values_table(df, na_name=True)



df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = df.loc[df["TotalCharges"].isnull(), "MonthlyCharges"]

df["TotalCharges"]

df["tenure"] = df["tenure"] + 1

df.isnull().sum()


#########################################
#FEATURE EXTRACTION
#########################################


##################################
# ÖZELLİK ÇIKARIMI
##################################

# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"


# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)


# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / df["tenure"]

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)


############################################
#ENCODING
###########################################

def rare_analyser(dataframe,col):
    print(pd.DataFrame({"value":dataframe[col].value_counts(),
                  "ratio":dataframe[col].value_counts()/len(dataframe)}))
    print("#"*60)


for col in cat_cols:
    rare_analyser(df,col)

num_cols,cat_cols,num_but_cat,cat_but_car=grab_col_names(df)



def label_encoder(dataframe,binary_col):
    label_encoder=LabelEncoder()
    label_encoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
for col in binary_cols:
    df=label_encoder(df,col)


df.head()


def one_hot_encoder(dataframe,categorical_cols):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=True,dtype=int)
    return dataframe

df=one_hot_encoder(df,cat_cols)
df.head()


##################################
# MODELLEME
##################################

df.head()
y = df["Churn_1"]
X = df.drop(["Churn_1","customerID"], axis=1)

models = [['LR', LogisticRegression(random_state=1)],
          ['KNN', KNeighborsClassifier()],
          ['CART', DecisionTreeClassifier(random_state=1)],
          ['RF', RandomForestClassifier(random_state=1)],
          ['XGB', XGBClassifier(random_state=1)],
          ["LightGBM", LGBMClassifier(random_state=1)],
          ["CatBoost", CatBoostClassifier(verbose=False, random_state=1)]]


for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"{name}")
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"Auc: {cv_results['test_roc_auc'].mean()}")
    print(f"Recall: {cv_results['test_recall'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"F1: {cv_results['test_f1'].mean().mean()}")

df.isnull().sum()









