# Not working


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

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
in_sample_acc = accuracy_score(y_train, lr.predict(X_train))

out_sample_acc = accuracy_score(y_test, lr.predict(X_test))

print(f'In-sample accuracy: {in_sample_acc:0.3}')
print(f'Out-of-sample accuracy: {out_sample_acc:0.3}')

for var, coef in zip(X_train.columns, lr.coef_[0]):
	print(f"{var:7} : {coef:+0.3}")

