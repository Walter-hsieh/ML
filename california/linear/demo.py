import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

california = fetch_california_housing()

X = pd.DataFrame(
	scale(california.data),
	columns = california.feature_names)

y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
	shuffle=True, 
	test_size=0.2,
	random_state=42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)


print(f"R2: {lm.score(X_train, y_train):0.3}")


import numpy as np
mean_y = np.mean(y_train)
squared_errors_mean = np.sum((y_train - mean_y)**2)
squared_errors_model = np.sum((y_train - lm.predict(X_train))**2)

R2 = 1 - (squared_errors_model/squared_errors_mean)

print(f"Computed R2: {R2}")


print([feat + ':' + str(round(coef, 1)) for feat, coef in zip(california.feature_names, lm.coef_)])


