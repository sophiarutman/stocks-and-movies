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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
