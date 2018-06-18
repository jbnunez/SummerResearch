#stock_cov_kmeans.py
import stock_data as data
import cov_k_means_class as cvk
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



model1 = cvk.spd_k_means(metric='aff')
model2 = cvk.spd_k_means(metric='leu')

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

model1.evaluate(X_test, y_test)
model2.evaluate(X_test, y_test)



