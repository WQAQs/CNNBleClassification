import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import globalConfig


TEST_SIZE = 0.1
RANDOM_STATE = 2019
best_accuracy = 0.0
NUM_CLASSES = 7
TrainFile = '.\\train_set.csv'
#TestFile = '.\\test_sample.csv'
BestModleFilePath = "best_model.h5"
TestFile = '.\\test_set.csv'
#BestModleFilePath = ".\\my_best_model\\dim_from_18points_7classes_mac\\18p_7c_with_all_data_2s.h5"
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# data preprocessing
def train_model_load_dataset(file):
    train_data = pd.read_csv(file)
    X = train_data.values[:, 2:]
    y = keras.utils.to_categorical(train_data.referencePoint_tag, NUM_CLASSES)
    # print(y)
    # y = train_data.values[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # print("X_train:{} , y_train:{}".format(X_train.shape, y_train.shape))
    # print("X_train: {}\n".format(X_train))
    # print("y_train: {}\n".format(y_train))
    # print("y_train: /n")
    # for df in y_train:
    #     print(df)
    # print("X_test:{} , y_test:{}".format(X_test.shape, y_test.shape))
    # !!!!!!must reshape the  X_train, X_test, y_train, y_test data to 3 dimensions!!!!!!!!
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_train = y_train.reshape((y_train.shape[0], NUM_CLASSES))
    y_test = y_test.reshape((y_test.shape[0], NUM_CLASSES))
    # print("reshaped X_train: {}\n".format(X_train))
    # print("reshaped y_train: {}\n".format(y_train))
    return X_train, y_train, X_test, y_test

def test_model_load_dataset(file):
    train_data = pd.read_csv(file)
    X = train_data.values[:,2:]
    y = keras.utils.to_categorical(train_data.referencePoint_tag, NUM_CLASSES)
    # print(y)
    # y = train_data.values[:,0]
   
    # print("X_train:{} , y_train:{}".format(X_train.shape, y_train.shape))
    # print("X_train: {}\n".format(X_train))
    # print("y_train: {}\n".format(y_train))
    # print("y_train: /n")
    # for df in y_train:
    #     print(df)
    # print("X_test:{} , y_test:{}".format(X_test.shape, y_test.shape))
    # !!!!!!must reshape the  X_train, X_test, y_train, y_test data to 3 dimensions!!!!!!!!
   
    X_test = X.reshape((X.shape[0], X.shape[1], 1))
    y_test = y.reshape((y.shape[0], NUM_CLASSES))
    # print("reshaped X_train: {}\n".format(X_train))
    # print("reshaped y_train: {}\n".format(y_train))
    return X_test, y_test

def evaluate_model(trainX, trainy, testX, testy):
    global best_accuracy
    verbose, epochs, batch_size = 0, 10, 32
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))  (x_train.shape[1],1)
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(trainX.shape[1], 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(trainy.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save(BestModleFilePath)
    return accuracy

def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores) # np.mean(): 求平均值   np.std()：求标准差
    print('summarize Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def run_experiment(repeats=10):
    # load train data
    trainX, trainy, testX, testy = train_model_load_dataset(TrainFile)
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
    print('best_score:%.3f \n' % (best_accuracy * 100))
   
def change_to_right(one_hot_labels):
    right_labels=[]
    for x in one_hot_labels:
        for i in range(0,NUM_CLASSES):
            if x[i]==1:
                # print("label_real:{}".format(i + 1))
                right_labels.append(i)
    return right_labels


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
   

    # run the experiment
    run_experiment()

    # load the best model and test
    testX, testy = test_model_load_dataset(TestFile)
    my_model = keras . models . load_model(BestModleFilePath)
    my_model.summary()
    preds = my_model . evaluate(testX, testy)
    print("my best model:{}".format(BestModleFilePath))
    print("test file = {}".format(TestFile))
    print("test result:")
    print ( "my_best_model Loss = " + str( preds[0] ) )
    print ( "my_best_model Test Accuracy = " + str( preds[1] ) )

    # show confusion_matrix
    labels_pred_all = my_model.predict_classes(testX)
    labels_all = change_to_right(testy)
    labels_all = pd.to_numeric(labels_all)
    print("labels_all:{}".format(labels_all))
    print("labels_all.dtype:{}".format(labels_all[0].dtype))
    print("labels_pred_all:{}".format(labels_pred_all))
    print("labels_pred_all.dtype:{}".format(labels_pred_all[0].dtype))
    confusion_matrix = confusion_matrix(labels_all, labels_pred_all)
    # confusion_matrix = tf.contrib.metrics.confusion_matrix(labels_pred_all, labels_all, num_classes=7, dtype=tf.int64, name=None, weights=None)
    # confusion_matrix = session.run(confusion_matrix)
    print("confucsion_matrix:{}".format(confusion_matrix))
    plt.matshow(confusion_matrix, cmap=plt.cm.gray)
    plt.show()
