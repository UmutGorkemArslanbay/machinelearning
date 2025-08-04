

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

data_path = ""

def main():
    # Veri yükleme
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop("medv", axis=1)
    y = df["medv"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alpha = 0.1
    # Bundan öncesine kadar max_iter yazmamıştım, bunun varsayılanı 1000'dir
    # Bazen yeterli olmayıp convergence warning almana yol açabilir bu yüzden iterasyon sınırını yükseltiyoruz ki model kesinlikle tam
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    
    print(f"Lasso (α={alpha}) → MSE: {mean_squared_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.2f}")
    
    coeffs = pd.Series(lasso.coef_, index=X.columns)
    print("\nKatsayılar (sıfır olmayanlar):")
    print(coeffs[coeffs!=0].sort_values(key=abs, ascending=False))
    
    # R² score 0.7 altında bu da 1/3'ünü açıklayamadığı anlamına geliyor.
    # Ayrıca kafadan ceza katsayısı belirlemek de yanlış, bunun en optimal en iyi değerini bulmamız gerekiyor bu data için.Bu konuya ileride değineceğim.
    
if __name__ == "__main__":
    main()
