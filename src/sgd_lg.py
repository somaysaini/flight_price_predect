import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time

from feautre_encoding import FlightData
data = FlightData()

X_train = data.x_train
y_train = data.y_train
X_test = data.x_test
Y_test = data.y_test


class SGDRegressor:
    
    def __init__(self,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):
        # init your coefs
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0,X_train.shape[0])
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_
                
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
        
        print(self.intercept_,self.coef_)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_




    # def fit(self, X_train, y_train):
    #     self.intercept_ = 0
    #     self.coef_ = np.ones(X_train.shape[1])
    #     n_samples = X_train.shape[0]
        
    #     for i in range(self.epochs):
    #         y_hat = np.dot(X_train, self.coef_) + self.intercept_
    #         error = y_train - y_hat
            
    #         intercept_der = -2 * np.mean(error)
    #         coef_der = -2 * np.dot(X_train.T, error) / n_samples
            
    #         self.intercept_ = self.intercept_ - (self.lr * intercept_der)
    #         self.coef_ = self.coef_ - (self.lr * coef_der)
            
    #     print(self.intercept_, self.coef_)

    # def predict(self,X_test):
    #     return np.dot(X_test,self.coef_) + self.intercept_
sgd = SGDRegressor(learning_rate=0.01,epochs=50)


start = time.time()
sgd.fit(X_train,y_train)
print("The time taken is",time.time() - start)

pred = sgd.predict(X_test)

print(r2_score(Y_test,pred))
