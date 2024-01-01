# Ideas  
1. 水文 , 土壤成分 -> crops recommendation or production analyze/prediction
2. 降雨量 & 收成分析

完全沒有做feature selection 直接Train:
Random Forest Regressor:  0.28320609694576533
Gradient Boost: 0.23827503542193295
SVR: about 0.3



一開始只有算年雨量， 年度平均氣溫，年度平均大氣壓，平均溫度
-> 後來加入了更詳細的數據: 相對溼度(%), 降水時數(hour), 降水日數(day)
找到更有相關性的feature.
原本年雨量的在feature selection 的時候只有 0.16778 的相關性, 使用年度降水時數就上升到了0.3887

做完Feature Selection後:
Random Forest Regressor: 0.07320186426779937
Gradient Boost: 0.08754802144067816
Support Vector Regressor ( with poly kernel): 0.09
Support Vector Regressor ( with sigmoid, rbf, linear): about 0.17


Testing different feature selection threshold -> 3 features works best (0.39)