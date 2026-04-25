import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X_train, X_test, y_train, y_test):
    plt.figure(figsize=(10, 6))
    x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.5, max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.5
    y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 0.5, max(X_train[:, 1].max(), X_test[:, 1].max()) + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', marker='o', edgecolor='k', s=50, label='Train Class 0')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red', marker='o', edgecolor='k', s=50, label='Train Class 1')
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='cyan', marker='^', edgecolor='k', s=80, label='Test Class 0')
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='yellow', marker='^', edgecolor='k', s=80, label='Test Class 1')
    plt.title(f'Decision Tree Boundary (Depth = {model.max_depth})')
    plt.xlabel('Feature X1')
    plt.ylabel('Feature X2')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()