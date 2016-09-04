from src.main.chapter2.GetTrainingData import GetTrainingData
from src.main.chapter2.Perceptron import Perceptron

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel('sepal length')
# plt.ylabel('petal length')
# plt.legend(loc='upper left')
# plt.show()
data = GetTrainingData()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(data.X, data.y)
# plt.plot(range(1, len(ppn.errors) + 1),
#          ppn.errors,
#          marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Misclassifications')
# plt.show()

plt = data.plot_decision_regions(ppn)
plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
