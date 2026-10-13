
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
matplotlib.use('QtAgg')




pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv("machine_learning/gen_ai_project/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("machine_learning/gen_ai_project/house-prices-advanced-regression-techniques/test.csv")
df = pd.concat([train, test],ignore_index=True)

df.head()
df = df.drop("index", axis=1)


######################################
# 1. Genel Resim
######################################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### Na #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)

check_df(df)



##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################


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


######################################
# 5. Korelasyon Analizi
######################################

corr = df[num_cols].corr()
corr

# Korelasyonların gösterilmesi
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()







######################################
# Görev 2 : Feature Engineering
######################################

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
    dataframe[variable] = dataframe[variable].astype("float64")
    df.loc[dataframe[variable] > up_limit, variable] = up_limit
    df.loc[dataframe[variable] < low_limit, variable] = low_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)


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

missing_values_table(df)

empty_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]



for col in empty_cols:
    df[col]=df[col].fillna("empty")


def missing_filling(data, target="SalePrice"):
    na_cols = [col for col in data.columns if data[col].isnull().sum() > 0]
    temp_target = data[target]

    num_cols, cat_cols, num_but_cat, cat_but_car = grab_col_names(df)

    num_cols = [col for col in num_cols if col != target]
    cat_cols = [col for col in cat_cols if col != target]

    # Sayısalları tek satırda medyanla doldur (döngüsüz)
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())

    # Kategorikleri tek satırda mod ile doldur (döngüsüz)
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

    # Hedef değişkeni eski haline getir
    data[target] = temp_target

    print(data[na_cols].isnull().sum())

    return data


df = missing_filling(df)
df.isnull().sum()






######################################
# yeni değişkenler oluşturunuz ve oluşturduğunuz yeni değişkenlerin başına 'NEW' ekleyiniz.
######################################


df["NEW_1st*GrLiv"] = (df["1stFlrSF"]+1)  * (df["GrLivArea"]+1)

df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])
"""
df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1) # 42
"""

# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"] # 32

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2 # 56

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF # 93

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF # 156

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF # 35


# Lot Ratio
df["NEW_LotRatio"] = df.GrLivArea / (df.LotArea +1)# 64

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / (df.LotArea+1) # 57

df["NEW_GarageLotRatio"] = df.GarageArea / (df.LotArea+1) # 69

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / (df.NEW_TotalHouseArea +1)# 36

# Dif Area
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF) # 73


df["NEW_OverallGrade"] = (df["OverallQual"]+1) * (df["OverallCond"] +1)# 61


df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt # 31

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt # 73

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd # 40

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt # 17

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd) # 30

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt # 48



drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# drop_list'teki değişkenlerin düşürülmesi
df.drop(drop_list, axis=1, inplace=True)


df.shape



######################################
# Label Encoding & One-Hot Encoding & Rare analizi yapınız ve rare encoder uygulayınız.
######################################



def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)



def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df,0.01)




cat_cols, cat_but_car, num_cols = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=True,dtype=int)
    return dataframe

df = one_hot_encoder(df, cat_cols)


df.head()
df.shape

##################################
# GÖREV 3: MODEL KURULUMU
##################################


train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]


y = train_df['SalePrice'] # np.log1p(df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)


models = [['LR', LinearRegression()],
          ['KNN', KNeighborsRegressor()],
          ['CART', DecisionTreeRegressor()],
          ['RF', RandomForestRegressor()],
          ['GBM', GradientBoostingRegressor()],
          ["XGBoost", XGBRegressor(objective='reg:squarederror')],
          ["LightGBM", LGBMRegressor()]]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")




##################
# hiperparametre optimizasyonlarını gerçekleştiriniz.
##################
import matplotlib
matplotlib.use('QtAgg')  # Alternative backend
import matplotlib.pyplot as plt
lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))  #rmse

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)



final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))


################################################################
# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=True):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:20])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)




