import pandas as pd

df = pd.read_csv('winequality-red.csv', sep=';')
print (df.describe())

import matplotlib.pylab as plt

plt.scatter(df['alcohol'],df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()

plt.scatter(df['volatile acidity'],df['quality'])
plt.xlabel('Volatile Acidity')
plt.ylabel('Quality')
plt.title('Volatile Acidity Against Quality')
plt.show()
                        
