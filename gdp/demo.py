import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# dataset got from these two websites
better_life_index = 'https://stats.oecd.org/index.aspx?DataSetCode=BLI'
dgp = 'https://www.imf.org/en/Publications/SPROLLS/world-economic-outlook-databases#sort=%40imfdate%20descending'


# prepare the data
df = pd.read_excel("life_statisfaction.xlsx")

print(df.columns)

X = np.array(df['gdp_per_capita']).reshape(-1,1)
y = np.array(df['life_satisfaction']).reshape(-1,1)


# visualize the data
fig, ax = plt.subplots()
ax.scatter(X, y)
ax.set_xlabel("GDP per Capita")
ax.set_ylabel("life satisfaction")
plt.ylim([0,10])
plt.xlim([0,100000])
# plt.savefig("GDP_vs_satisfaction.png", dpi=1000)

# select a linear model
model = sklearn.linear_model.LinearRegression()

# Train model
model.fit(X, y)

# Make a prediction for Japan
X_new = [[48812.76]] # Japan's gpd per capita
print(model.predict(X_new)) # output: 6.561/ actual value is 6.1

# draw fitting line
X_fit = np.linspace(0,10**5,10**5).reshape(-1,1) 
y_fit = model.predict(X_fit)

ax.plot(X_fit, y_fit,color='red')
plt.savefig('linear_regression.png', dpi=1000)
plt.show()










