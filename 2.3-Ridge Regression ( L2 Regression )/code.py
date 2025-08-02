

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

data_path = ""

def main():
    # Veriyi yükleme
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop("medv",axis=1)
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # tanim.md'de açıklandı.
    scaler = StandardScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    alpha = 1 # tanımda bahsedilen ceza katsayısı
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled,y_train)
    
    y_pred = ridge.predict(X_test_scaled)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² :", r2_score(y_test, y_pred))
    
    coeffs = pd.Series(ridge.coef_, index=X.columns)
    print("Özellik katsayıları:\n", coeffs.sort_values(key=abs, ascending=False))
    
    # R² score 0.7 altında ala 1/3'ünü açıklayamadığı anlamına geliyor.
    # Ayrıca kafadan ceza katsayısı belirlemek de yanlış, bunun en optimal en iyi değerini bulmamız gerekiyor bu data için.
    
    

if __name__ == "__main__":
    main()
    