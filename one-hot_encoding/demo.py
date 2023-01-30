from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()
enc = OneHotEncoder(sparse=False)

qualitative = ['red', 'red', 'green', 'blue', 'red', 'blue', 'blue', 'green']

labels = lbl.fit_transform(qualitative).reshape(-1,1)

print(labels)

# print(enc.fit_transform(labels))



