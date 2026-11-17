import matplotlib
import io
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]
print("Rows:", len(homes))

# (a) Fit on all 414 rows
model = LinearRegression().fit(X, y)
print(f"Slope (price units per 100 m): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# (b) Predict at 500 m (distance_100m = 5)
new_home = pd.DataFrame({"distance_100m": [5]})
pred = model.predict(new_home)[0]
print(f"Predicted price per unit area at 500 m: {pred:.2f}")

# (c) Plot observations and fitted line
xs = pd.DataFrame({"distance_100m": [X["distance_100m"].min(), X["distance_100m"].max()]})
plt.figure(figsize=(8, 5))
plt.scatter(X["distance_100m"], y, s=18, alpha=0.6, label="Observations")
plt.plot(xs["distance_100m"], model.predict(xs), color="red", linewidth=2, label="Fitted line")
plt.xlabel("Distance to nearest MRT station (100 m)")
plt.ylabel("House price per unit area (10,000 NTD/ping)")
plt.title("Task 1: Housing Price vs. Distance to Metro")
plt.legend()
plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150)
with open("task1_housing.png", "wb") as f:
    f.write(buf.getvalue())
