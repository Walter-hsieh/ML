import pandas as pd

filename = 'https://github.com/lmassaron/datasets/'
filename += 'releases/download/1.0/tennis.feather'
tennis = pd.read_feather(filename)

X = tennis[['outlook', 'temperature', 'humidity', 'wind']]
X = pd.get_dummies(X)
y = tennis.play

	
dt = DecisionTreeClassifier()
dt.fit(X, y)

import dtreeviz
viz_model = dtreeviz.model(dt, X, y, 
	target_name='play_tennis',
	feature_names=X.columns,
	class_names=["No", "Yes"])

v = viz_model.view()

v.show()

v.save("tennis_play.svg")