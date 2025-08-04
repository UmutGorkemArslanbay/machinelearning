
import pandas as pd
import numpy as np
import xgboost as xgb # XGBoost için
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data_path = ""

def main():
    # Veri yükleme
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop("medv", axis=1)
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DMatrix aşaması var, bu veriyi işlemeyi ciddi bir şekilde hızlandırır.
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective" : "reg:squarederror", # Modelin optimizasyon hedefini belirtir.
        "eval_metric" : "rmse", # Eğitim sırasında izlenecek değerlendirme metriğini Root Mean Squared Error olarak seçtik.
        "eta" : 0.1, # Learning Rate 
        "max_depth" : 5,
        "subsample" : 0.8, # Birden fazla ağaç kuracağı için bi rastgelelik olması gerekiyor ki ezber olmasın veya aynı yanlışlıklar olmasın bu yüzden train verisinin %80 'ini kullanacak. Ensemble çeşitliliği artar.
        "colsample_bytree" : 0.8, # Her ağacı kurarken features'da %80 'ini rastgele seçer.
        "lambda" : 1.0, # Ridge Regression katsayısı
        "alpha" : 0.0 # Lasso Regression katsayısı
    }

    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,  # Maksimum oluşturulacak ağaç sayısıdır. (tur sayısı)
        evals=[(dtrain, "train"), (dtest, "valid")],
        early_stopping_rounds=20, # Validasyon seti 20 turun sonunda gelişmezse durdurur.
        verbose_eval=50 # Her 50 turda bir ekrana log basar, True tüm turlarda log bastırır.
    )
    
    y_pred = model.predict(dtest)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("RMSE   :", rmse)
    print("R²     :", r2_score(y_test, y_pred))

    # R² score 0.91 geldi iyi bir sonuç
    # RMSE 2.52, modelin test setindeki ortalama tahmin hatası 2.52 civarıdır.
    
if __name__ == "__main__":
    main()

    