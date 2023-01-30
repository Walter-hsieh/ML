from sklearn.linear_model import LinearRegression
import numpy as np

y = np.array([0,0,0,0,1,1,1,1])
X = np.array([1,2,3,4,5,6,7,8]).reshape(8,1)

lm = LinearRegression()
lm.fit(X, y)
preds = lm.predict(X)

for y_true, y_pred in zip(y, preds):
	print(f"{y_true} -> {y_pred:+0.3}")
