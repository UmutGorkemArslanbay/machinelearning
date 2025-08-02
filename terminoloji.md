# Algoritma
 Bir problemi çözmek için izlenen adımlar dizisidir.
# Model
 Bir algoritmanın veri üzerinde eğitilmesi sonucu oluşan matematiksel temsildir.
# Feature 
 Modelin girdi olarak kullandığı bireysel özelliklerdir.
# Dataset
 Modelin öğrenimi için kullandığımız verisetidir.
# Training
 Modelin verilerden öğrenmesi sürecidir.
# Prediction 
 Eğitilmiş bir modelin yeni veriler üzerinden tahmin yapmasıdır.
# Overfitting
 Modelin eğitim verilerine aşırı uyum sağlaması, genellikle yeteneğini kaybetmesi,ezberlemesidir.
# Underfitting
 Modelin hem train hem de test verilerinde kötü performans göstermesidir.
# Bias
 Modelin sistematik hata yapma eğilimidir.
# Variance
 Modelin farklı eğitim setlerinden ne kadar farklı sonuçlar ürettiğidir.
 - Elimizde [2,4,6,8] değerleri var. Bu değerlerin aralarında ne kadar "dağıldığını" ölçmek için önce ortalamasını alırız.
 - Ortalama bu veride 5
 # Total Sum of Squares (SStot) ya da Total Variance
 - Sonra her bir değerin ortalamadan farkını kareye alır ve toplarız.Bu veri seti için  SStot = 20
 # Residual
  - Gerçek değerler ile tahmin aradasındaki farklardır.Bu veri seti için arasınaki farklar [−1,−1,−1,−1]
 - Bu örnek için de modelimiz [3,5,7,9] tahminlerini yaptığını varsayalım.
 # Residual Sum of Squares (SSres) ya da Explained Variance
 - Gerçek değerler ile tahminler arasında farkların karelerini alıp toplarız. SSres = 4
 # Korelasyon
  İki değişkenin arasındaki ilişkinin yönünü ve gücünü ölçen istatiksel bir ölçümdür.