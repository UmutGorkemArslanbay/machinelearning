

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

data_path = ""

def main():
    # Veri yükleme
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop("medv", axis=1)
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=100, # Ormanda oluşturulacak ağaç sayısı 
        # Hemen akıllara bir soru bu ne kadar çok olursa o kadar iyi değil mi ?
        # n_estimators arttırıldıkça modelin varyansı azalır, dolayısıyla doğruluk bir süre için iyileşir.
        # Burada her bir ağaç bir yük ve zaman demektir.
        # İlk ağaçlar ile yüzlerce ağaç arasında model hızla yükselebilir lakin bir müddet sonra elde edinilen kazanım çok küçük olur.
        # Sorumuz teoride doğru lakin verimsizdir.
        max_depth=None, # Sınırsız derinliği temsil eder.
        min_samples_leaf=5,
        max_features="sqrt", # features alt küme ( her bölmede sqrt kullanılacak ).
        bootstrap= True,
        random_state=42,
    )
    
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"RandomForest → MSE: {mse:.2f}, R²: {r2:.2f}")

    # Regression öğrenirken 0.81 yakaladık, hala iyileşmek için optimizasyon yapılabilir buna değineceğim.Bunun için acele etmeyelim.

if __name__ == "__main__":
    main()
    