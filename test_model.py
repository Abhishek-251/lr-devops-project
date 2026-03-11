import numpy as np
from sklearn.linear_model import LinearRegression

X=np.array([[1],[2],[3],[4]])
y=np.array([2,4,6,8])

def test_coefficient():
    model=LinearRegression()
    model.fit(X,y)
    assert round(model.coef_[0], 2) == 2.00, "Coefficient should be 2.00"

def test_intercept():
    model=LinearRegression()
    model.fit(X,y)
    assert round(model.intercept_, 2) == 0.00, "Intercept should be 0.00"

def test_prediction():
    model=LinearRegression()
    model.fit(X,y)
    prediction=model.predict([[5]])
    assert round(prediction[0], 2) == 10.00, "Prediction for input 5 should be 10.00"