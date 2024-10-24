# Load necessary libraries
library(MASS)
library(glmnet)
library(caret)
library(randomForest)
library(rpart)
library(neuralnet)

# Load the Boston dataset
data(Boston)

# Inspect the dataset
head(Boston)

# Create a linear regression model to predict the median value of owner-occupied homes
model <- lm(medv ~ ., data = Boston)

# View summary information about the model
summary(model)

# Create a scatterplot to visualize the relationship between the actual and predicted values
plot(Boston$medv, fitted(model), xlab = "Actual", ylab = "Predicted")
abline(lm(fitted(model) ~ Boston$medv), col = "red")

# Calculate the root mean squared error (RMSE) of the model
actual <- Boston$medv
predicted <- predict(model, newdata = Boston)
RMSE <- sqrt(mean((actual - predicted)^2))
cat("The RMSE of the linear regression model is", round(RMSE, 2), "\n")

# Split the data into training and testing sets
set.seed(123)
train_index <- sample(nrow(Boston), 0.8 * nrow(Boston))
train_data <- Boston[train_index, ]
test_data <- Boston[-train_index, ]

# Ridge regression model 
x_train <- as.matrix(train_data[, -14])
y_train <- train_data[, 14]
cv_fit <- cv.glmnet(x_train, y_train, alpha = 0, standardize = TRUE, type.measure = "mse")
lambda_optimal <- cv_fit$lambda.min

# Fit ridge regression model to the training data
ridge_fit <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_optimal, standardize = TRUE)

# Predict house prices for the test data 
x_test <- as.matrix(test_data[, -14])
y_test <- test_data[, 14]
y_pred_ridge <- predict(ridge_fit, newx = x_test)

# Compute RMSE for ridge regression
rmse_ridge <- sqrt(mean((y_test - y_pred_ridge)^2))
cat("The RMSE of the ridge regression model is", round(rmse_ridge, 2), "\n")

# Plot predicted vs actual for ridge regression
plot(y_test, y_pred_ridge, main = "Ridge Regression: Actual vs Predicted House Prices", xlab = "Actual Prices", ylab = "Predicted Prices")
abline(0, 1, col = "red")

# Fit a random forest regression model to the training data
rf_model <- randomForest(medv ~ ., data = train_data, ntree = 500)

# Make predictions on the testing data
rf_predictions <- predict(rf_model, newdata = test_data)

# Calculate RMSE for random forest
rf_rmse <- sqrt(mean((rf_predictions - y_test)^2))
cat("The RMSE of the random forest model is", round(rf_rmse, 2), "\n")

# Plot predicted vs actual for random forest
plot(y_test, rf_predictions, main = "Random Forest: Actual vs Predicted House Prices", xlab = "Actual Prices", ylab = "Predicted Prices")
abline(0, 1, col = "red")

# Fit a decision tree regression model to the training set
dt_model <- rpart(medv ~ ., data = train_data)

# Predictions on the testing set using decision tree
dt_predictions <- predict(dt_model, newdata = test_data)

# Calculate RMSE for decision tree
dt_rmse <- sqrt(mean((dt_predictions - y_test)^2))
cat("The RMSE of the decision tree model is", round(dt_rmse, 2), "\n")

# Plot predicted vs actual for decision tree
plot(y_test, dt_predictions, main = "Decision Tree: Actual vs Predicted House Prices", xlab = "Actual Prices", ylab = "Predicted Prices")
abline(0, 1, col = "red")

# Train a KNN regression model on the training set
knn_model <- train(medv ~ ., data = train_data, method = "knn", trControl = trainControl(method = "cv"), tuneLength = 10)

# Use the trained KNN model to predict house prices on the testing set
knn_predictions <- predict(knn_model, test_data)

# Calculate RMSE for KNN
knn_rmse <- sqrt(mean((knn_predictions - y_test)^2))
cat("The RMSE of the KNN model is", round(knn_rmse, 2), "\n")

# Plot predicted vs actual for KNN
plot(y_test, knn_predictions, main = "KNN: Actual vs Predicted House Prices", xlab = "Actual Prices", ylab = "Predicted Prices")
abline(0, 1, col = "red")

# Combine Linear Regression and Random Forest models' predictions for comparison
lm_predictions <- predict(lm(medv ~ ., data=train_data), test_data)
lm_rmse <- sqrt(mean((test_data$medv - lm_predictions)^2))

cat("The RMSE of the linear regression model is", round(lm_rmse, 2), "\n")

# Combine RMSE values into a data frame.
rmse_df <- data.frame(Model=c("Linear Regression", "Random Forest", "Ridge Regression", "Decision Tree", "KNN"),
                      RMSE=c(lm_rmse, rf_rmse, rmse_ridge, dt_rmse, knn_rmse))

print(rmse_df)
