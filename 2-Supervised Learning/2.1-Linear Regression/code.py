# Elimizdeki dataset Boston.csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data_path = "" # Dataset'in yolunu giriyoruz.

def main():
    # Veri yükleme
    df = pd.read_csv(data_path, index_col=0)
    
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

    # R² score 0.67, 0.7 altında ala 1/3'ünü açıklayamadığı anlamına geliyor.
    # MSE 24.29,
    
    
    
if __name__ == "__main__":
    main()