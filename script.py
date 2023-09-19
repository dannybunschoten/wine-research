import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


plt.hist(data[data["wine type"] == 0]["pH"], bins=15)
plt.title("histogram of pH values in red wine")

plt.hist(data[data["wine type"] == 1]["pH"], bins=15)
plt.title("histogram of pH values in white wine")

plt.show()

def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x-mu)/ sigma) ** 2))

red_wine_ph_mean = data[data["wine type"] == 0]["pH"].mean()
print(f"Mean of pH in red wine: {red_wine_ph_mean}")

white_wine_ph_mean = data[data["wine type"] == 1]["pH"].mean()
print(f"Mean of pH in white wine: {white_wine_ph_mean}")

red_wine_ph_std = data[data["wine type"] == 0]["pH"].std()
print(f"Standard deviation of pH in red wine: {red_wine_ph_std}")

white_wine_ph_std = data[data["wine type"] == 1]["pH"].std()
print(f"Standard deviation of pH in white wine: {white_wine_ph_std}")

x_values = np.linspace(2.6, 3.75, 400)

likelyhood_Class_1 = gaussian_pdf(x_values, red_wine_ph_mean, red_wine_ph_std)
likelyhood_Class_2 = gaussian_pdf(x_values, white_wine_ph_mean, white_wine_ph_std)

plt.plot(x_values, likelyhood_Class_1, label="P(pH | Red Wine)")
plt.plot(x_values, likelyhood_Class_2, label="P(pH | White Wine)")

plt.legend()
plt.title("Likelihood function for Red and White Wine based on pH level")
plt.xlabel("pH")
plt.ylabel("Probability")
plt.show()