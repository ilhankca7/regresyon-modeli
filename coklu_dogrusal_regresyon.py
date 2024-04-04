import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
data = pd.read_csv("C:/Users/ilhan/Downloads/Advertising.csv")
veri = data.copy()
veri.drop(columns=["Unnamed: 0"],axis=1,inplace=True)

y=veri["Sales"]
X = veri.drop(columns="Sales", axis=1)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
model=lr.fit(X_train,y_train)


def skor(model,x_train,x_test,y_train,y_test):
    egitimtahmin = model.predict(x_train)
    testtahmin = model.predict(x_test)
    
    r2_egitim=mt.r2_score(y_train,egitimtahmin)
    r2_test = mt.r2_score(y_test,testtahmin)
    
    mse_egitim = mt.mean_squared_error(y_train,egitimtahmin)
    mse_test = mt.mean_squared_error(y_test,testtahmin)
    
    return[r2_egitim,r2_test,mse_egitim,mse_test]
    

sonuc1 = skor(model=lr,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)

print("Eğitim R2={} Eğitim MSE={}".format(sonuc1[0],sonuc1[2]))
print("Test R2={} Test MSE={}".format(sonuc1[1],sonuc1[3]))


lr_cv = LinearRegression()
k = 5
iterasyon = 1
cv = KFold(n_splits=k)

for egitim_index, test_index in cv.split(X):
    X_train, X_test = X.iloc[egitim_index], X.iloc[test_index]
    y_train, y_test = y.iloc[egitim_index], y.iloc[test_index]
    lr_cv.fit(X_train, y_train)
    
    y_train_pred = lr_cv.predict(X_train)
    y_test_pred = lr_cv.predict(X_test)
    
    egitim_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    egitim_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    print("İterasyon: {}".format(iterasyon))
    print("Eğitim R2 Score: {:.4f}, Eğitim MSE: {:.4f}".format(egitim_r2, egitim_mse))
    print("Test R2 Score: {:.4f}, Test MSE: {:.4f}".format(test_r2, test_mse))
    
    iterasyon += 1
      
                                                        
