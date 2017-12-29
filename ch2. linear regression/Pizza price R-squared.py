from sklearn.linear_model import LinearRegression
import numpy as np
# Training data
X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
X_test = [[8],  [9],   [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]] 
# Create and fit the model
model = LinearRegression()
model.fit(X,y)
print('R-squared: %.4f' % model.score(X_test,y_test))
