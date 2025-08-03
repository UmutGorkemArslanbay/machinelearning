
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, r2_score

data_path = ""

def main():
    # Veri yükleme
    
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop("medv", axis=1)
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = DecisionTreeRegressor(
        max_depth=3, # Bu maksimum bölünmeyi, maksimum derinliği (katman sayısını) sınırlar.
        # max_depth=3 için örnek mesela 
        # 2 olsaydı ptratio bölünmesi olmayacaktı
        # 4 olsaydı başka bir özellik eklenecek ptratio altında
        
        # |--- rm <= 6.941 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |--- lstat <= 9.295 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |--- ptratio <= 18.400 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |   |--- value: [18.50] Buradaki değerlerin ortalaması -> Leaf
        # |   |   |--- ptratio >  18.400 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |   |--- value: [20.30] Buradaki değerlerin ortalaması -> Leaf
        # |   |--- lstat >  9.295 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |--- dis <= 2.155 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |   |--- value: [11.75] Buradaki değerlerin ortalaması -> Leaf
        # |   |   |--- dis >  2.155 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |   |--- value: [ 8.60] Buradaki değerlerin ortalaması -> Leaf
        # |--- rm >  6.941 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |--- dis <= 2.179 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |--- tax <= 296.000 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |   |--- value: [24.50] Buradaki değerlerin ortalaması -> Leaf
        # |   |   |--- tax >  296.000 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |   |--- value: [22.40] Buradaki değerlerin ortalaması -> Leaf
        # |   |--- dis >  2.179 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |--- lstat <= 4.950 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz. 
        # |   |   |   |--- value: [36.10] Buradaki değerlerin ortalaması -> Leaf
        # |   |   |--- lstat >  4.950 -> min_samples_leaf burada kontrol edilir kaç adet değer var diye bu örnek için 5'den azsa bu bölünme olmaz.
        # |   |   |   |--- value: [31.80] Buradaki değerlerin ortalaması -> Leaf

        min_samples_leaf=5,# Her leaf en az bulunması gereken örnek sayısını ifade eder. Eğer bir yaprakta bu sayıdan daha az veri varsa bölünme gerçekleşmez.Yukarıda da örnekte gösterdim ne manaya geldiğini.
        random_state=42, # Bunu zaten ifade etmiştim, bir kez daha ifade ediyim.Bu deterministik hale getirir.Aynı verisetiyle ve aynı değerlerle sonucun aynı olmasını sağlıyor bizlere.
        )
    
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    print("DecisionTree → MSE: %.2f, R²: %.2f" % (
    mean_squared_error(y_test, y_pred),
    r2_score(y_test, y_pred)
    ))
    
    # Ağacın bölünmesini metin olarak yazdırmak istersek
    tree_rules = export_text(dt, feature_names=list(X.columns))
    print("\n=== Tree Structure ===")
    print(tree_rules)
    
    # R² score 0.7 nin üstüne çıktı ama hala iyileştirme yapılabilir.
    # Ayriyetten kafadan max_depth, min__samples_leaf belirlemek de yanlış, bunun en optimal en iyi değerini bulmamız gerekiyor bu data için.Bu konuya ileride değineceğim.
    
if __name__ == "__main__":
    main()