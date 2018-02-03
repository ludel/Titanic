import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

for dataset in combine:
    dataset.loc[dataset['Age'].isnull(), 'Age'] = 0
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 1
    dataset.loc[(dataset['Age'] > 16) & (
                dataset['Age'] <= 30), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 30) & (
                dataset['Age'] <= 60), 'Age'] = 3
    dataset.loc[dataset['Age'] > 60, 'Age'] = 4

    dataset.loc[dataset['Fare'] < 7, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] >= 7) & (dataset['Fare'] <= 14), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] >= 14) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = 0
    dataset.loc[dataset['Embarked'] == "C", 'Embarked'] = 1
    dataset.loc[dataset['Embarked'] == "Q", 'Embarked'] = 2
    dataset.loc[dataset['Embarked'] == "S", 'Embarked'] = 3
    dataset['seul'] = 0
    dataset.loc[dataset['SibSp'] + dataset['Parch'] == 0, 'seul'] = 1

train_df = train_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)

X_train = train_df.drop(['Survived','PassengerId'], axis=1).values
Y_train = train_df['Survived'].values
X_test = test_df.drop('PassengerId', axis=1).values

learning_rate = 0.001
training_epochs = 150
display_step = 5

number_of_inputs = 8
number_of_outputs = 1

nbr_layers = 4

with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, 50], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[50], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[50, 60], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[60], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[60, 50], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[50], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[50, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print("Starting training epochs : ", training_epochs)
    for epoch in range(training_epochs):
        session.run(optimizer, feed_dict={X: X_train, Y: Y_train})
        print(round((epoch*100)/training_epochs, 0), "%")

    final_training_cost = session.run(cost, feed_dict={X: X_train, Y: Y_train})

    print("Final Training cost: {}".format(final_training_cost))

    Y_predicted_scaled = session.run(prediction, feed_dict={X: X_test})
    print("Prediction finale :", Y_predicted_scaled)

print("==== Fin ====")
