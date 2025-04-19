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
# Install if needed
# install.packages("mgcv")

library(mgcv)

# Load and clean data
df <- read.csv("merged_data/merged.csv")
df <- subset(df, (GOOG == 1 | T == 1 | XOM == 1 | SPX == 1 | LDOS == 1) & Year > 1998 & Year < 2019)

# Fit GAM with splines on numeric predictors
gam_model <- gam(Box.Office ~ 
                   s(Open) + s(High) + s(Adj.Close) + s(Volume) + 
                   s(Year) + s(Runtime) + s(Metascore) + s(IMDB.Votes) +
                   GOOG + T + XOM + SPX + LDOS,
                 data = df)

# Summary
summary(gam_model)

# Predict and evaluate
preds <- predict(gam_model, newdata = df)

# R² manually
r2 <- 1 - sum((df$Box.Office - preds)^2) / sum((df$Box.Office - mean(df$Box.Office))^2)
cat("R² Score:", r2, "\n")

#
#
#
library(e1071)
library(caret)  # for preprocessing
library(Metrics) 

# Load dataset
df <- read.csv("merged_data/merged.csv")

# Filter data
df <- subset(df, (GOOG == 1 | T == 1 | XOM == 1 | SPX == 1 | LDOS == 1) & Year > 1998 & Year < 2019)

# Select features and target
features <- c("Open", "High", "Adj.Close", "Volume", "Year", "Runtime", 
              "Metascore", "IMDB.Votes", "GOOG", "T", "XOM", "SPX", "LDOS")

df <- df[complete.cases(df[, c(features, "Box.Office")]), ]  # remove NAs

X <- df[, features]
y <- df$Box.Office

# Scale features
preproc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preproc, X)

# Split into training/testing
set.seed(42)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X_scaled[train_idx, ]
X_test  <- X_scaled[-train_idx, ]
y_train <- y[train_idx]
y_test  <- y[-train_idx]

# Fit SVR model
svr_model <- svm(x = X_train, y = y_train, type = "eps-regression", kernel = "radial")

# Predict
y_pred <- predict(svr_model, X_test)

# Evaluate
mse_val <- mse(y_test, y_pred)
r2_val <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

cat("SVR Mean Squared Error:", round(mse_val, 2), "\n")
cat("SVR R² Score:", round(r2_val, 4), "\n")
#
#
#
