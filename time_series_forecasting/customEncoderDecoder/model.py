import tensorflow as tf
import numpy as np
import os
import math
import random
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

tfk = tf.keras
tfkl = tf.keras.layers

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/hw2_model_v8.0.3.h5'))

    def predict(self, X):
        window = 400
        stride = 1
        telescope = 1 
        to_predict = 864

        col_max = np.max(X, axis=0)
        col_min = np.min(X, axis=0)
        X_norm = np.divide(X - col_min, col_max - col_min)
        X = X_norm[-window:]
        X = np.expand_dims(X, axis=0)

        out = np.array([])
        
        for i in range(int(math.ceil(to_predict / stride))):
            pred = self.model.predict(X)
            pred = np.array(pred)

            X = np.concatenate((X[0], pred[0][:stride]), axis=0)
            X = np.expand_dims(X, axis=0)
            if i != 0:
                out = np.concatenate((out, pred[0][:stride]), axis=0)
            else:
                out = pred[0][:stride]


            X = X[0][-window:]
            X = np.expand_dims(X, axis=0)

        out = out[:to_predict]
        out = np.multiply(out, col_max - col_min)
        out = out + col_min
        out = tf.convert_to_tensor(out)
        return out
