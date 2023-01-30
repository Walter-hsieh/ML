import pandas as pd

filename = 'https://github.com/lmassaron/datasets/'
filename += 'releases/download/1.0/titanic.feather'
titanic = pd.read_feather(filename)

from sklearn.model_selection import train_test_split

X = titanic.iloc[:,:-1]
y = titanic.iloc[:,-1]

(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0, shuffle=True)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(min_samples_split=5)

dt.fit(X_train, y_train)

accuracy = dt.score(X_test, y_test)

print(f"test accuracy: {accuracy:0.3}")

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=90)
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o',
drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha")
# plt.savefig("result.png", dpi=1000)
plt.show()

best_pruning =list()
for ccp_alpha in ccp_alphas:
	if ccp_alpha > 0:
		dt = DecisionTreeClassifier(random_state=0,
			ccp_alpha=ccp_alpha)
		dt.fit(X_train, y_train)
		best_pruning.append([ccp_alpha, dt.score(X_test, y_test)])

best_pruning = sorted(best_pruning, key=lambda x: x[1], reverse=True)

best_ccp_alpha = best_pruning[0][0]
dt = DecisionTreeClassifier(random_state=0,
	ccp_alpha=best_ccp_alpha)

dt.fit(X_train, y_train)
accuracy = dt.score(X_test, y_test)

print(f"test accuracy: {accuracy:0.3}")

print("Number of nodes in the last tree is: {} with ccp_alpha: {:0.3}".format(dt.tree_.node_count, best_ccp_alpha))


import dtreeviz

viz_model = dtreeviz.model(dt, X, y,
	target_name='survived',
	feature_names=X.columns,
	class_names=["No", "Yes"])

v = viz_model.view()

v.show()

v.save("titanic.svg")

