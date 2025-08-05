


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet

data_path = "" # Dataset'in yolunu giriyoruz.

alpha_enet = 0.1 # ElasticNet toplam ceza gücünü belirleyen parametredir.
l1_ratio = 0.5 # 0 %100 Ridge cezası demek.
               # 1 %100 Lasso cezası demek.
               # 0.2 %20 Lasso %80 Ridge demek.
               # 0.8 %80 Lasso %20 Ridge demek.

def main():
    # Veri yükleme
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop("medv", axis=1)
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    enet = ElasticNet(
        alpha=alpha_enet,
        l1_ratio = l1_ratio,
        max_iter = 10000,
        random_state=42,
    )
    enet.fit(X_train_scaled, y_train)
    y_pred = enet.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"ElasticNet (α={alpha_enet}, l1_ratio={l1_ratio}) → MSE: {mse:.2f}, R²: {r2:.2f}")


    coeffs = pd.Series(enet.coef_, index=X.columns)
    print("\nÖzellik katsayıları (mutlak büyüklüğe göre):")
    print(coeffs.sort_values(key=abs, ascending=False))
    
    # R² score 0.7 altında bu da 1/3'ünü açıklayamadığı anlamına geliyor.
    # Ayrıca kafadan ceza katsayısı belirlemek de yanlış, bunun en optimal en iyi değerini bulmamız gerekiyor bu data için.Bu konuya ileride değineceğim.
    
    
if __name__ == "__main__":
    main()
    
    