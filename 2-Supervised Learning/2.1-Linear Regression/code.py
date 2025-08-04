# Elimizdeki dataset Boston.csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data_path = ""

def main():
    
    # Datayı ilk başta okuyalım
    df = pd.read_csv(data_path, index_col=0)
    # print(df.head(5)) bunu etkinleştirerek okuyabilirsiniz veya csv'yi açıp bakabilirsiniz.
    
    X = df.drop("medv", axis=1) # medv sütununu kaldırdı
    y = df["medv"] # medv sütunun y'ye taşıdı.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train,y_train)
    

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    print(f"Linear Regression sonuçları:")
    print(f"  • MSE: {mse:.2f}")
    print(f"  • R² : {r2:.2f}\n")
    
    # Burada tahmin etmeye çalıştığımız şey, fiyat ve pozitif katsayılar fiyatı arttırıcı, negatifler azaltıcı bir etkiye sahiptir.
    coeffs = pd.Series(model.coef_, index=X.columns)
    print("Özellik katsayıları (en yüksekten en düşüğe):")
    print(coeffs.sort_values(ascending=False))
    
    # Bunu grafiğe dökecek olursak
    plt.figure(figsize=(6,6))
    plt.scatter(y_test,y_pred,edgecolors="k")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Gerçek medv")
    plt.ylabel("Tahmin medv")
    plt.title("Real vs Predict")
    plt.tight_layout()
    plt.show()


    # Fotoğrafdaki kırmızı çizgi x ve y eşitlendiğinde noktalar bu çizgi üzerinde olur.
    # Model ne kadar iyiyse noktalar bu çizgi etrafında yoğunlaşır.
    # Model orta kısımlarda iyi bir sonuç göstermesine rağmen uçlardaki varyansyonları iyi yakalayamamış.
    # Bu hatayı gidermek için daha farklı yöntemler kullanılabilir.
    
    
if __name__ == "__main__":
    main()