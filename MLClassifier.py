import os
import numpy as np
from os import listdir
from skimage import io
from skimage.transform import resize
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# calculating metrics for a neural network model
from keras import backend as k


###############  M O D E L   E V A L U A T I O N   M E T R I C S   #################
def recall_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + k.epsilon()))


# settings
img_size = 100
num_class = 10
test_size = 0.2
validation_size = 0.25


def get_img(data_path):
    # getting image array from path
    img = io.imread(data_path, as_gray=True)
    img = resize(img, (img_size, img_size))
    return img


def get_dataset(dataset_path='Dataset'):
    # Getting all data from the datapath
    try:
        X = np.load('npy.dataset/X.npy')
        Y = np.load('npy.dataset/Y.npy')
    except:
        labels = listdir(dataset_path)  # Getting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path + '/' + label
            for data in listdir(datas_path):
                img = get_img(datas_path + '/' + data)
                X.append(img)
                Y.append(int(label))

        # Create dataset
        X = 1 - np.array(X).astype('float32') / 255
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)

        if not os.path.exists('npy_dataset/'):
            os.makedirs('npy_dataset/')
        np.save('npy_dataset/X.npy', X)
        np.save('npy_dataset/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X, X_test, Y, Y_test


# get data set (80% training , 20% testing)
x_train, x_test, y_train, y_test = get_dataset()
# split training set (60% training , 20% validation)

x_train = x_train.reshape(x_train.shape[0], 100, 100, 1)
X_test = x_test.reshape(x_test.shape[0], 100, 100, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=1)

print("training shape", x_train.shape)

print("testing shape", x_test.shape)

print("validation shape", x_val.shape)

print("model1 ==> 1 layers , 256 neuron  ")


# define baseline model
def model1():
    # create model 2 layers , 64 neuron each
    model = Sequential([Dense(64, activation='relu', input_shape=(100, 100, 1)),
                        Dense(64, activation='relu')
                        ])

    model.add(Flatten())
    model.add(Dense(num_class, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])
    return model


# define baseline model
def model2():
    # create model 7 layers , 32 neuron each
    model = Sequential([Dense(32, activation='relu', input_shape=(100, 100, 1)),
                        Dense(32, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(32, activation='relu'),
                        ])

    model.add(Flatten())
    model.add(Dense(num_class, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])
    return model


def model3():
    # create model 3 layers , 128 neuron each
    model = Sequential([Dense(128, activation='relu', input_shape=(100, 100, 1)),
                        Dense(128, activation='relu'),
                        Dense(128, activation='relu'),
                        ])

    model.add(Flatten())
    model.add(Dense(num_class, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])
    return model


def model4():
    # create model 1 layers , 256 neuron each
    model = Sequential([Dense(256, activation='relu', input_shape=(100, 100, 1))
                        ])

    model.add(Flatten())
    model.add(Dense(num_class, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])
    return model


# evaluate a model using k-fold cross validation
def evaluate_model(model, dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kFold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kFold.split(dataX):
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit the model
        history = model.fit(trainX, trainY, validation_data=(x_val, y_val), epochs=5, batch_size=200)
        # evaluate the model
        loss, acc, f1_score, precision, recall = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # store scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


# build the model
model = model1()
# evaluate model
evaluate_model(model, x_train, y_train, 5)
