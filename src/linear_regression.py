import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('data/wine.csv')

x = df[['alcohol']]
y = df[['quality']]

model = linear_model.LinearRegression()
model.fit(x,y)

print("model:", model.coef_)
print("model:",model.intercept_)

plt.figure(figsize=(10,6))

plt.scatter(x,y)
plt.xlabel("alcohol")
plt.ylabel("quality")

plt.plot(x, model.predict(x))
plt.grid(True)
plt.show()
