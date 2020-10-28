import numpy as np
import pandas as pd

def next_batch(training_data,batch_size,steps):
    batch_index = np.random.randint(0, len(training_data) - steps)
    y = training_data.iloc[batch_index:batch_index + steps + 1].as_matrix().reshape(1, steps + 1)
    return y[:,:-1].reshape(-1, steps, 1), y[:,1:].reshape(-1, steps, 1)

def mape(y_pred, y_test):
    return 100*np.mean(abs(y_pred - y_test)/y_test)
