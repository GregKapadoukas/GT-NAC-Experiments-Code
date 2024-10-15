import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score

def cnn_classifier(data_train, categories_train, data_test, categories_test):
    categories_train = pd.get_dummies(categories_train)
    categories_list = np.unique(categories_test.to_numpy())
    categories_test = pd.get_dummies(categories_test)
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], 1)
    data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], 1)
    model = keras.Sequential(
        [
            layers.Conv1D(32, (3,), activation='relu', input_shape=(data_train.shape[1],1)),
            layers.MaxPooling1D((2,)),
            layers.Conv1D(64, (3,), activation='relu'),
            layers.MaxPooling1D((2,)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(categories_train.shape[1], activation='softmax')
        ]
    )

    if(categories_train.shape[1] == 2):
        model.compile(
            loss = keras.losses.BinaryCrossentropy(),
            optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.6, beta_2=0.999),
            metrics = ["binary_crossentropy", "categorical_accuracy"]
        )

    else:
        model.compile(
            loss = keras.losses.CategoricalCrossentropy(),
            optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.6, beta_2=0.999),
            metrics = ["categorical_crossentropy", "categorical_accuracy"]
        )

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.fit(data_train, categories_train, batch_size=32, epochs=10, validation_split=0.2, callbacks=[callback], verbose=2)
 
    categories_pred = model.predict(data_test)
    categories_pred = [categories_list[np.argmax(p)] for p in categories_pred]
    categories_test = categories_test.to_numpy()
    categories_test = [categories_list[np.argmax(p)] for p in categories_test]
    accuracy = accuracy_score(categories_test, categories_pred)
    macro_precision, macro_recall, macro_fscore, macro_support = score(categories_test, categories_pred, average='macro')
    print(f'Accuracy: {accuracy}')
    print(f'Macro-Precision: {macro_precision}')
    print(f'Macro-Recall: {macro_recall}')
    print(f'Macro-F-Score: {macro_fscore}')
    print(classification_report(categories_test, categories_pred, zero_division=1))
    return({"Accuracy":accuracy, "Macro-Precision":macro_precision, "Macro-Recall":macro_recall, "Macro-F-Score":macro_fscore})
