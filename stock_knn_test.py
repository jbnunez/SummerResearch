#stock_knn_test.py
import stock_data as data
import cov_knn as ck
import numpy as np

X = data.X_covs
y = data.labels
y_len = y.shape[0]
test_len = y_len//5
train_len = y_len - test_len

shuffled = np.random.permutation(zip(X, y))
test = shuffled[:test_len]
train = shuffled[test_len:]
X_test, y_test = zip(*test)
X_train, y_train = zip(*train)


model1 = ck.spd_knn(X_train, y_train, k=5, metric='aff')
model2 = ck.spd_knn(X_train, y_train, k=5, metric='leu')

model1.evaluate(X_test, y_test)
model2.evaluate(X_test, y_test)








