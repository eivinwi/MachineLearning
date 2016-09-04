import numpy as np

from src.main.chapter2.AdalineGD import AdalineGD
from src.main.chapter2.GetTrainingData import GetTrainingData

data = GetTrainingData()

# Standardized
X_std = np.copy(data.X)
X_std[:, 0] = (X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
X_std[:, 1] = (X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, data.y)

data.X = X_std
plt = data.plot_decision_regions(ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost) + 1),
         ada.cost,
         marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
