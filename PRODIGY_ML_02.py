import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create the dataset (square footage, number of bedrooms, price)
# Dataset format: [Square Footage, Number of Bedrooms, Price]
data = np.array([
    [1400, 3, 245000], [1600, 3, 312000], [1700, 3, 279000], [1875, 4, 308000],
    [1100, 2, 199000], [1550, 3, 219000], [2350, 4, 405000], [2450, 4, 324000],
    [1425, 3, 319000], [1700, 3, 255000], [1750, 3, 299000]
])

# Step 2: Split the data into independent variables (X) and dependent variable (y)
X = data[:, 0:2]  # Square footage and number of bedrooms
y = data[:, 2]    # Price

# Add an extra column of ones to X (to account for the intercept term in the model)
m = len(y)
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Step 3: Apply the Normal Equation to compute the optimal parameters (theta)
def normal_equation(X, y):
    # Formula: theta = (X.T * X)^(-1) * X.T * y
    return np.linalg.inv(X.T @ X) @ X.T @ y

theta = normal_equation(X, y)

# Step 4: Use the trained model (theta) to predict house prices
def predict(square_footage, num_bedrooms):
    # Prediction formula: price = theta0 + theta1 * square_footage + theta2 * num_bedrooms
    return theta[0] + theta[1] * square_footage + theta[2] * num_bedrooms

# Test predictions on a few houses
print("Predicted price for a 1500 sq ft, 3-bedroom house: $", predict(1500, 3))
print("Predicted price for a 2000 sq ft, 4-bedroom house: $", predict(2000, 4))

# Step 5: Visualize the model's performance (only for square footage for simplicity)
plt.scatter(data[:, 0], y, color='blue', label='Actual Prices')
plt.plot(data[:, 0], X @ theta, color='red', label='Predicted Prices')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()
plt.show()
