######################################################
# Sales Prediction with Linear Regression
######################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Qt5Agg')

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("ders_materyalleri/machine_learning-220803-231749/machine_learning/datasets/advertising.csv")
df.shape

X = df[["TV"]]
y = df[["sales"]]


##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]


##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T


# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},  #sns.regplot linear regression modelini görselleştirmek için
                ci=False, color="r")  #color="" model çizgi rengi
#scatter_kws={'color': 'b', 's': 9} veri noktaları için rengi mavi büyüklüğü 9
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")  #ekrana başlık ekler
g.set_ylabel("Satış Sayısı")   #x eksrninin ismi
g.set_xlabel("TV Harcamaları")  #x eksrninin ismi
plt.xlim(-10, 310)  #grafikte x eksenindeki aralık
plt.ylim(bottom=0)   #grafikte y eksenin min 0 olsun eğer top limit yazılmazsa verilere bakar top limiti seçer
plt.show()


##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X)  #bulduğun katsayıyı test seti ile çarparak tahmin sonuçlarını verir
mean_squared_error(y, y_pred)
# 10.51
y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE
reg_model.score(X, y)
#Regresyon modellerinde bağımsız değişkenlerin bağımlı değişkendeki değişkenliğin yüzde kaçını açıklayabildiğini gösterir.
#İşte hiçbir bağımsız değişken kullanmadan, sadece ortalamayı tahmin olarak söylemeye "Hiç model kurmamak" (Naive / Baseline Model) denir.
#R^2 yok ettiğin hatanın yüzdesidir:Başlangıçtaki toplam hata: 25 Modelden sonra kalan hata: 10 Modelin yok ettiği (açıkladığı) hata: 25 - 10 = 15
#yüzde olarak 15/25 =0.60 Satışlardaki artış ve azalışların %60'ını TV harcamalarıyla açıklayabiliyorsun demektir.
# Kalan %40'lık kısım ise TV harcaması dışındaki faktörlere (hava durumu, indirimler, şans vb.) bağlıdır.
#O 25 sayısı, verdiğin ilk örnekteki standart sapmanın karesinden geldi.Matematikte bir verinin standart sapmasının karesi o verinin Varyansını verir.
#verinin standart sapması 5 ise, hiçbir veri kullanmadan sadece ortalamayı tahmin ederek yapacağın karesel hata (MSE) kesinlikle 25 olur.
######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("ders_materyalleri/machine_learning-220803-231749/machine_learning/datasets/advertising.csv")


X = df.drop('sales', axis=1)

y = df[["sales"]]


##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_


##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# 0.0468431 , 0.17854434, 0.00258619

# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

yeni_veri = [[30], [10], [40]]     #bu tek sütun 3 satır olur [[30, 10, 40]]bu  3 sütun tek satır
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test RKARE
reg_model.score(X_test, y_test)


# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71


#parametreleri bulmanın 2 yöntemi vardır analitik çözüm ve gradient descent sklearn varsayılan olarak analitik çözüm kullanır ama
#analitik çözümde tersini alma işlemi olduğu için bu işlem veri fazlaysa uzun sürebilir bu yüzden adım adım paramtre güncellmesi yapılan
# gradient descent daha mantıklıdır.

######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)










