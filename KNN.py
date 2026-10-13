################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("ders_materyalleri/machine_learning-220803-231749/machine_learning/datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)
#algoritma şöyle çalışır en yakın x adet gözlemi bulur öklid algoritması ile eğer sınıflandırma ise x adet gözlemden frekansı en çok olanı seçer
#eğer regresyon ise x adet gözlemin ortalamsı 

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74
# AUC
roc_auc_score(y, y_prob)
# 0.90

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# 0.73
# 0.59
# 0.78

# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()  #varsayılan hyperparameter değerlerini verir

knn_params = {"n_neighbors": range(2, 50)}


#GridSearchCV verdiğin tüm hiperparametre kombinasyonlarını dener ve sana en iyi olan sonucu verir
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,  #işlemciyi tam performans kullan
                           verbose=1).fit(X, y)   #verbose=1 sounç raporu bekliyorsan onu verir

knn_gs_best.best_params_   #en iyi sonuc yakınlığın 17 olduğu

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)
#knn_model nesne olduğu için knn_model(**knn_gs_best.best_params_) gibi bi şey yapılımaz nesneye parantez açılmaz ya KNeighborsClassifier(**knn_gs_best.best_params_)
# yapcan ya da knn_model.set_params(**knn_gs_best.best_params_) set_params sklearnin fonksiyonu
#knn_model.set_params(**knn_gs_best.best_params_) ** yazınca python arka planda bunu  knn_model.set_params(n_neighbors=11) bu hale getirir



cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)  #predictte de aynı şekilde tek tek öklid ile uzaklık bulur en yakın olanı tahmn eder











