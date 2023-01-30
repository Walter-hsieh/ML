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
print(f"R2_lm: {lm.score(X_train, y_train):0.3}")



from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
poly_X_train = pf.fit_transform(X_train)
lm = LinearRegression()
lm.fit(poly_X_train, y_train)
r2 = lm.score(poly_X_train, y_train)
print(f"R2_pf: {r2:0.3}")











