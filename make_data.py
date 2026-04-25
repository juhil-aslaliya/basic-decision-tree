# import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

def get_data():
    X, y = make_circles(n_samples=500, noise=0.2, factor=0.0, random_state=4269)
    return train_test_split(X, y, test_size=0.2, random_state=42) 
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='class 0', alpha=0.7)
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='class 1', alpha=0.7)
# plt.title("Inbuilt make moons dataset")
# plt.xlabel("Feature X1")
# plt.ylabel("Feature X2")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()