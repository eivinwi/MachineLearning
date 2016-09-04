import matplotlib.pyplot as plt
import numpy as np

from src.main.chapter2.AdalineGD import AdalineGD
from src.main.chapter2.GetTrainingData import GetTrainingData

data = GetTrainingData()
n_iters = 10

fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
X = np.copy(data.X)
y = np.copy(data.y)
ada1 = AdalineGD(n_iter=n_iters, eta=0.01).fit(X, y)

ax[0].plot(range(1, len(ada1.cost) + 1),
           np.log10(ada1.cost),
           marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

X = np.copy(data.X)
y = np.copy(data.y)
ada2 = AdalineGD(n_iter=n_iters, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost) + 1),
           np.log10(ada2.cost),
           marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
