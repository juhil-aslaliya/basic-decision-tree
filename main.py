from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from make_data import get_data
from models import DecisionTree
from visualisation import plot_decision_boundary

X_train, X_test, y_train, y_test, features, result = get_data()

tree = DecisionTree(max_depth=6, features=features, result_name=result, result_vals={0: 'No', 1: 'Yes'})
tree.fit(X_train, y_train)

print('--- Decision Tree Structure ---')
tree.print_tree()
print('-------------------------------\n')

predictions = tree.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy on unseen data: {accuracy*100:.2f}%')

# plot_decision_boundary(tree, X_train, X_test, y_train, y_test)