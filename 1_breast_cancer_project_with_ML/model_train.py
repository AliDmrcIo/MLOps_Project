"""
1.Dosya: Data Science ve ML işlemleri

Problem: Göğüs kanseri var mı yok mu anlamak

Veri seti: Scikit-Learn'de bulunan breast_cancer verisetidir
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import joblib

# verisetini çağır ve bağımlı bağımsız değişkenleri ayır
data = load_breast_cancer()
X = data.data   # bağımsız değişkenlerimiz - independent variables - features
y = data.target # bağımlı değişkenimiz - dependent variable - label

# train test split ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model create/build
model = RandomForestClassifier(n_estimators=100, random_state=42)

# model training
model.fit(X_train, y_train)

# train accuracy checking
train_accuracy = model.score(X_train, y_train) # train veriseti için prediction yapıp hata karelere bakıyor
print(f"Train Accuracy: {train_accuracy:.2f}")

# test accuracy checking
test_accuracy = model.score(X_test, y_test) # test veriseti için prediction yapıp hata karelere bakıyor
print(f"Test Accuracy: {test_accuracy:.2f}")

# modeli kaydet : endpoint içerisine yükleyebilmek için kaydediyoruz
joblib.dump(model, 'breast_cancer_model.pkl') # dosyanın adı 'breast_cancer_model.pkl' olsun