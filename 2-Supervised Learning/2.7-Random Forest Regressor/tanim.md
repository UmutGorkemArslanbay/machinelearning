# Random Forest Regressor
 Aynı veri üzerinde tek bir güçlü ama değişken model kurmak yerine (bahsi geçen regressor Decision Tree) birden fazla karar ağacı oluşturulur, orman gibi.
 Her ağaç bootstrap yöntemiyle seçilen bir alt küme üzerinden eğitilir.
 Tek bir karar ağacı, her düğümde bütün features'ların en iyi split ederse;
 -Ağaçlar birbirine çok benzer olur.
 -Çeşitlilik artmaz.
 -Overfitting avantajını kaybederiz.
 Bunu engellemek için Random Forest Regressor features için rastgele alt kümeler oluşturur.Bu durum da ağaçlar arası çeşitliliği arttırır.
 

