import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.main.chapter2.Adaline import Adaline

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)
df.tail()
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
n_iters = 10

fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = Adaline(n_iter=n_iters, eta=0.01).fit(X, y)

ax[0].plot(range(1, len(ada1.cost) + 1),
           np.log10(ada1.cost),
           marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = Adaline(n_iter=n_iters, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost) + 1),
           np.log10(ada2.cost),
           marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
