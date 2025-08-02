# Elimizdeki dataset Boston.csv

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures # Polynomial Regression için import etmemiz gereken kod
from sklearn.linear_model import LinearRegression # sonradan Linear Regression yapacağımız için import etmemiz gereken kod
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data_path = ""

def main():
    # Veriyi yükleyelim
    df = pd.read_csv(data_path, index_col=0)
    # index_col = 0 indexi 1 den başlamasına yarar. 0'ı kaldırır.
    X = df.drop("medv",axis=1)
    y = df["medv"]

    # Şimdi polynomial regression için özellikleri gireceğiz.
    poly = PolynomialFeatures(degree=2, include_bias=False)
    # degree = 2, girdileri karesiyle girmesini sağlar 3 yazsaydık küpüyle girmesini sağlardı
    # include_bias ise sabit terim (bias, yani “1”) sütununu eklemez.
    # Burada akıllara şu soru geliyor. 
    # Eklenirse ne olur ?
    # Eklenmezse ne olur ?
    # Ne zaman true ne zaman false olmalı ?
    
    # Bu sütun linear modellerde intercept (kesişim) terimini temsil eder.
    # Çoğu scikit-learn modeli ( Linear Regression, Ridge, Lasso ) varsayılan olarak fit_intercept = True ile gelir, bu ifade modelin kendi içinde bir sabit terim öğrenmesi anlamına gelir ve böyle durumlarda include_bias = False kullanılır
    # Ama model fit_intercept = False ile kurduysanız veyahut da hiç desteklemeyen bir algoritmaya sahipte, o zaman sabit terimi siz sağlamak zorundasınız include_bias = True kullanılır.
    
    X_poly = poly.fit_transform(X) # Normalde fit terimleri nasıl üretmesini sağlar transform da öğrendiği kurala göre genişletilmiş bir matrise çevirir, İkisinin birleşimi de öğrenme,dönüştürme fit_transform'dur.
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Polynomial Regression (degree=2)")
    print(f" • MSE: {mse:.2f}")
    print(f" • R² : {r2:.2f}\n")
    
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, edgecolor='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Gerçek medv")
    plt.ylabel("Tahmin edilen medv")
    plt.title("Real vs Predict")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

