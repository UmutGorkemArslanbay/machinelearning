# Decision Tree Regressor 
 Ağaç tabanlı en temel regresyon yöntemidir.
 Bir "karar ağacı" inşa ederek her yaprakta (leaf) o yaprağa düşen eğitim örneklerinin hedef değerlerinin ortalamasını tahmin olarak kullanır.
 # Mantık
  Başlangıçta tüm veriler kökte (root) toplanır.
  Her düğümde (node) bir özellik ve bir threshold (eşik) seçilerek veriyi iki parçaya ("sol" ve "sağ") böler.
  Amaç, her iki parçanın içinde hedef değerlerin varyansını en çok düşüren bölünmeyi bulmak.

  Bunu şimdi daha örnekleştirelim
  5 adet medv değerleri olsun elimde [10,12,11,29,31]
  Ortalama 18.6
  Parent Varyans = 87.44 civarı
  threshold = 20 olsun

  Sol parçayı medv < 20 sağ parçası 20 > medv olarak düşünelim

  Sol parça için [10,12,11]       Sağ parça için [29,31]
  Ortalamas = 11                  Ortalaması = 30
  Varyans = 0.67                  Varyans = 1


  Şimdi ağırlıklı ortalama varyansı hesaplayalım
  Sol parça 3 örnek Sağ parça 2 örnek Toplam = 5
  VarSol = 0.67 VarSağ = 1
  Split Varyans = 3/5 x 0.67 + 2/5 x 1 = 0.80
  
  Parent Varyans = 87.44
  Split Varyans = 0.80
  Varyans azalması = 86.64

  Şimdi de threshold = 11 yapalım

  Sol parça için [10,11]       Sağ parça için [12,29,31]
  Ortalamas = 10.5             Ortalaması = 24
  Varyans = 0.25               Varyans = 72.67

  Split Varyans = 3/5 x 0.25 + 2/5 x 72.67 = 43.70
  Varyans azalması 43.74

  Bu senaryo için threshold 20 > threshold 11 daha sağlıklı bir model oluyor.

