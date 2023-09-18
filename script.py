import numpy as np
import pandas as pd

white = pd.read_csv("winequality-white.csv", sep=";")
white["wine type"] = 1
red = pd.read_csv("winequality-red.csv", sep=";")
red["wine type"] = 0
white = white.sample(n=len(red), random_state=42)
data = pd.concat([white, red], ignore_index=True)


print(data.head())
print(data.shape[1])
data.drop(labels=["volatile acidity", "quality", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "alcohol", "sulphates", "fixed acidity", "density"], 
          axis=1, inplace=True, errors="ignore")

print(data.head())

import matplotlib.pyplot as plt

plt.scatter(data["pH"], data["wine type"])
plt.xlabel("pH")
plt.ylabel("wine type")
plt.show()

def gaussian(x, )
