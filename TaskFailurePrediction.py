import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

train_df=pd.read_csv('train_data.csv')

train_df.info()

print(train_df.shape)

train_df = train_df.dropna(axis=0)

"""Reshape X_train and X_test to 3D since Conv1D requires 3D data"""

X = train_df.iloc[:, 1:5]
y = train_df.iloc[:,5]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train = X_train.values
X_test = X_test.values

X_train = X_train.reshape(-1, X_train.shape[1],1)
X_test = X_test.reshape(-1, X_test.shape[1],1)

print(X_train.shape)
print(X_test.shape)

"""Convert the Target label to categorical"""

target_train = y_train
target_test = y_test
Y_train=to_categorical(target_train)
Y_test=to_categorical(target_test)

print(Y_train.shape)
print(Y_test.shape)

"""Performance Evaluation Function"""

def showResults(test, pred):
    accuracy = accuracy_score(test, pred)
    precision=precision_score(test, pred, average='weighted')
    f1Score=f1_score(test, pred, average='weighted')
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    cm=confusion_matrix(test, pred)
    print(cm)

"""# **Hybrid CNN LSTM**"""



import tensorflow as tf
tf.keras.backend.clear_session()

model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=64,kernel_size=5,strides=1,padding="causal",activation="sigmoid",input_shape=(X_train.shape[1],X_train.shape[2])),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="sigmoid"),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="sigmoid"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="sigmoid"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2)
])
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.7),
              metrics=['acc'])
model.summary()

history = model.fit(X_train, Y_train,epochs=15)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png', format='png', dpi=1200)
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png', format='png', dpi=1200)
plt.show()

predictions = model.predict(X_test, verbose=1)

predictcv=np.argmax(predictions,axis=1)
actual_valuecv=np.argmax(Y_test,axis=1)
showResults(actual_valuecv, predictcv)

hyd = accuracy_score(actual_valuecv, predictcv)
f1hyd=f1_score(actual_valuecv, predictcv, average='weighted')

"""#Hidden Markov Model"""

!pip install hmmlearn

from hmmlearn.hmm import GaussianHMM
hmm = GaussianHMM(n_components=2)
hmm.fit(X)
predictions = hmm.predict(X)
print("*Confusion Matrix for HMM: ")
print(confusion_matrix(y, predictions))

print("*Classification report for HMM: ")
print(classification_report(y, predictions))

val3 = accuracy_score(y, predictions) *100
print(val3)

f1hmm = f1_score(y, predictions, average='weighted')
f1hmm

"""#Bi-LSTM"""

from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.layers.core import Activation
import tensorflow as tf

es=EarlyStopping(patience=7)
model=Sequential()
model.add(Bidirectional(LSTM(12,input_shape=(X_train.shape[1],X_train.shape[2]))))
model.add(Dense(units=2))
model.add(Activation('sigmoid'))

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.6, momentum=0.8),
              metrics=['acc'])

history = model.fit(X_train, Y_train,epochs=20)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png', format='png', dpi=1200)
plt.show()


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png', format='png', dpi=1200)
plt.show()

predictions = model.predict(X_test, verbose=1)

predictcv=np.argmax(predictions,axis=1)
actual_valuecv=np.argmax(Y_test,axis=1)
showResults(actual_valuecv, predictcv)

bilstm = accuracy_score(actual_valuecv, predictcv)
f1bilstm=f1_score(actual_valuecv, predictcv, average='weighted')

bilstm

f1bilstm

def predict_task_status(test_data):
  result=model.predict(test_data)
  res=[]
  print(result)
  for data in result:
    if data[0]<data[1]:
      res.append("Successful")
    else:
      res.append("Failed")
  return res

# [ [memory_GB], [network_log10_MBps], [local_IO_log10_MBps] , [NFS_IO_log10_MBps] ]
test_data=[
    [[44.3904],[-1.0262],[0.8033],[-3.0000]],
    [[31.5839],[-1.4608],[-0.6080],[-2.9967]],
    [[154.4610],[-0.5508],[-0.3637],[-3.0000]],
    [[32.4488],[1.8770],[-0.1212],[-3.0000]],
    [[5.3928],[0.1131],[0.2250],[-3.0000]],
    [[4.67],[-1.2131],[-1.12],[-3.0000]]
    ]

predict_task_status(test_data)
