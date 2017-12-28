from sklearn.linear_model import LinearRegression
import numpy as np
# Training data
X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
# Create and fit the model
model = LinearRegression()
model.fit(X,y)
print ('A 12" pizza should cost: $%.2f' % model.predict(np.array([12]).reshape(1,-1))[0])
