

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


data_path = r"C:\Users\umut\Desktop\machinelearning\machinelearning\2-Supervised Learning\2.5-Logistic Regression\data\Boston.csv"

def main():
    # Veri yükleme
    df = pd.read_csv(data_path, index_col = 0)
    # High Price sütunu ekledik ve 21 den büyükse 1 aksi 0 boolen bir yapı kurduk.
    df["high_price"] = (df["medv"] >= 21).astype(int)
    
    X = df.drop(["medv","high_price"], axis=1)
    y = df["high_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, 
    stratify= y # Bu y'nin train ve test için aynı oranda 0 ve 1 elde etmesini sağlar.
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        penalty= "l2", # Ridge Regression
        C = 1,         # Ceza gücü (küçük -> sıkı , büyük -> esnek)
        solver="liblinear",
        # Optimize edici seçmemizi sağlar ( algoritma bir nevi)
        # liblinear : One-vs-rest (binary class), küçük ve orta boy veri setleri için geliştirilmiş, penalty = l1 ve l2 uygun.
        # lblfgs : Softmax (multinominal), küçük ve orta boy veri setleri için geliştirilmiş, penalty = l1 uygun.
        # sag : Softmax, büyük veri setleri için geliştirilmiş, penalty = l2 uygun.
        # saga : Softmax, büyük veri setleri için geliştirilmiş, penalty = l1 ve l2 uygun.
        # newton-cg : Softmax, küçük ve orta boy veri setleri için geliştirilmiş, penalty = l2 uygun.
        random_state=42,
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Low", "High"])

    print(f"Accuracy : {acc:.3f}")
    print(f"ROC AUC  : {auc:.3f}\n")
    print("Classification Report:")
    print(report)
    
    
    # Accuracy 0.873, Test setindeki 102 örneğin 89’unu doğru sınıflandırmışız. Yani %87,3 oranında “High”/“Low” fiyatı doğru tahmin ediyoruz.
    
    # ROC AUC 0.932, Model pozitif (“High”) ve negatif (“Low”) sınıfları ayırt etme konusunda çok güçlü.
    # 1 Mükemmel ayırma, 0.5 rastgele tahmin.


    
    
if __name__ == "__main__":
    main()