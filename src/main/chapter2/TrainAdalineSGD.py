from src.main.chapter2.AdalineSGD import AdalineSGD
from src.main.chapter2.GetTrainingData import GetTrainingData

data = GetTrainingData()

n_iter = 15
eta = 0.01
random_state = 1

ada = AdalineSGD(n_iter=n_iter, eta=eta, random_state=random_state)
ada.fit(data.X, data.y)
plt = data.plot_decision_regions(ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost) + 1),
         ada.cost,
         marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()