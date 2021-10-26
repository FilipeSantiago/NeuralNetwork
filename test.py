from layer.dense import Dense
from activation.relu import Relu
from activation.softmax import Softmax
from sklearn.model_selection import train_test_split
from neural_network.sequential import Sequential
import numpy as np
from sklearn.metrics import accuracy_score
import pandas

df = pandas.read_csv("data/iris.data", names=['a', 'b', 'c', 'd', 'class'], header=None)
df = df.sample(frac=1).reset_index(drop=True)

train_df = df[:75]
test_df = df[75:150]

data, data_test, _y, _y_test = train_test_split(df[['a', 'b', 'c', 'd']], df['class'], test_size=0.2, random_state=0)

y = pandas.get_dummies(_y, prefix='class')
y_test = pandas.get_dummies(_y_test, prefix='class')

learning_rate = 0.1

nn = Sequential(Dense(4, 100, Relu()),
                Dense(100, 3, Softmax()))

nn.start_learning(data, y, 100, learning_rate=np.e**-2, batch_size=32)

res = nn.predict(data_test)

y_pred = np.argmax(res, axis=1)
y_true = np.argmax(y_test.values, axis=1)

print(y_pred)
print(y_true)
print(accuracy_score(y_true, y_pred))
