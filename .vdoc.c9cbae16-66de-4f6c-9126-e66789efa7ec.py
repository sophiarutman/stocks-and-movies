# type: ignore
# flake8: noqa
#
import pandas as pd
import numpy as np 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Load and filter dataset
df = pd.read_csv('merged_data/merged.csv')
df = df[(df["GOOG"] == 1) | (df["T"] == 1) | (df["XOM"] == 1) | (df["SPX"] == 1) | (df["LDOS"] == 1)]
df = df[(df["Year"] > 1998) & (df["Year"] < 2019)]

# Feature selection
X = df[["Open", "High", "Adj Close", "Volume", "Year", "Runtime", "Metascore", "IMDB Votes", "GOOG", "T", "XOM", "SPX", "LDOS"]]
y = df["Box Office"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and fit model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
#
#
#
import pandas as pd
import numpy as np 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Load and filter dataset
df = pd.read_csv('merged_data/merged.csv')
df = df[(df["GOOG"] == 1) | (df["T"] == 1) | (df["XOM"] == 1) | (df["SPX"] == 1) | (df["LDOS"] == 1)]
df = df[(df["Year"] > 1998) & (df["Year"] < 2019)]
# Feature selection
X = df[["Year", "Runtime", "Metascore", "IMDB Votes"]]
y = df["Box Office"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and fit model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
#
#
#
#
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load and filter dataset
df = pd.read_csv('merged_data/merged.csv')
df = df[(df["GOOG"] == 1) | (df["T"] == 1) | (df["XOM"] == 1) | (df["SPX"] == 1) | (df["LDOS"] == 1)]
df = df[(df["Year"] > 1998) & (df["Year"] < 2019)]

# Feature selection
X = df[["Open", "High", "Adj Close", "Volume", "Year", "Runtime", "Metascore", "IMDB Votes", "GOOG", "T", "XOM", "SPX", "LDOS"]]
y = df["Box Office"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create spline transformer with degree=3 (cubic), 5 knots
spline = SplineTransformer(degree=3, n_knots=5, include_bias=False)

# Combine spline transformation and regression in a pipeline
model = make_pipeline(StandardScaler(), spline, LinearRegression())

# Fit and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Sample degrees and colors for plotting
degrees = [2, 3, 13]
colors = ['blue', 'red', 'green']

plt.figure(figsize=(10, 6))

# Plot the original data
plt.scatter(X, y, color="gray", label="Avg Box Office Data", s=60)

# Loop through degrees and fit polynomial models
for deg, color in zip(degrees, colors):
    # Transform features with polynomial degree
    poly = PolynomialFeatures(degree=deg)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)  # Notice the use of transform here instead of fit_transform

    # Fit the model
    model = LinearRegression().fit(X_poly_train, y_train)
    
    # Predict using the model
    y_pred = model.predict(X_poly_test)
    
    # Calculate R²
    r2 = r2_score(y_test, y_pred)  # Use y_test, not y, for R² on test data
    
    # Plot the predictions
    plt.plot(X_poly_test, y_pred, color=color, linewidth=2, label=f"Degree {deg} (R² = {r2:.3f})")

# Labeling the axes and title
plt.xlabel("Average Stock Adj Close ($)")
plt.ylabel("Average Box Office Revenue ($)")
plt.title("Polynomial Regression of Box Office Revenue vs. Stock Market (1999–2019)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
#
#
#
