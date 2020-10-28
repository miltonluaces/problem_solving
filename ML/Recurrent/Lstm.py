
from ML.forecasting import *
from ML.utils import next_batch, mape
import logging
import os
import time

def LSTM_sequence(batch_size = 1, num_time_steps = 12, num_neurons = 100,
                  learning_rate = 0.03, num_iterations = 2000, pred_length = 12):
    logger = logging.getLogger()
    logger.info('LSTM Basic Sequence starting. Following hyperparameters are given:{0},{1},{2},{3},{4},{5}'.format(batch_size, num_time_steps , num_neurons,
                  learning_rate, num_iterations,pred_length))
    milk_prod = pd.read_csv('./Data/MilkProd.csv', index_col= 'Date')
    logger.info('Data ingestion complete. Running the LSTM Basic Sequence model')
    milk_prod.index = pd.to_datetime(milk_prod.index)
    # train test split 
    # functionalize
    N, D = milk_prod.shape
    train_N = N - pred_length
    logger.info('Using the first {0} records for training, {1} records for testing.'.format(train_N, pred_length))
    train_x = milk_prod.head(train_N) 
    test_x = milk_prod.tail(pred_length)
    
    scaler = MinMaxScaler()
    train_x = pd.DataFrame(scaler.fit_transform(train_x), columns = train_x.columns, index = train_x.index)
    test_x = pd.DataFrame(scaler.transform(test_x), columns = test_x.columns, index = test_x.index)
    logger.info('Used MinMaxScaling from sklearn as preprocessing')
    tf.reset_default_graph()
    """
    num_time_steps = 12
    num_neurons = 100
    num_outputs = 1
    learning_rate = 0.03
    num_iterations = 2000
    batch_size = 1
    """
    
    X = tf.placeholder(tf.float32, shape = [None, num_time_steps, batch_size])
    y = tf.placeholder(tf.float32, shape = [None, num_time_steps, batch_size])
    
    cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.LSTMCell(num_units = num_neurons, activation = tf.nn.relu), 
                                                  output_size = batch_size)
    
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
    
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss)
    logger.info('Single layer LSTM Code with {0} neurons created'.format(num_neurons))
    
    init = tf.global_variables_initializer()
    
    
    saver = tf.train.Saver()
    
    # remove the previous model run
    try:
        shutil.rmtree('./basic_lstm_model')
    except:
        pass
    
    t0 = time.time()
    with tf.Session() as sess:
        sess.run(init)
        
        for iteration in range(num_iterations):
            
            X_batch, y_batch = next_batch(train_x, batch_size, num_time_steps)
            
            sess.run(train, feed_dict = {X: X_batch, y: y_batch})
    
            if iteration % 100 == 0:
                
                mse = loss.eval(feed_dict = {X:X_batch, y: y_batch})
                print(iteration, '\tMSE', mse)
        #os.mkdir('basic_lstm_model')
        saver.save(sess, "./basic_lstm_sequence_model/model")
    
    t1 = time.time()
    logger.info('LSTM training completed in: {0} seconds. Starting predictions'.format(t1 - t0))
    
    with tf.Session() as sess:
        
        # Use your Saver instance to restore your saved rnn time series model
        saver.restore(sess, "./basic_lstm_sequence_model/model")
        
        train_seed = list(train_x[-pred_length:].as_matrix())
        
        for iteration in range(pred_length):
            X_batch = np.array(train_seed[-num_time_steps:]).reshape(1,num_time_steps, 1)
            
            y_pred = sess.run(outputs, feed_dict = {X:X_batch})
            
            train_seed.append(y_pred[0,-1,0])
    
    
    
    predictions = scaler.inverse_transform(np.array(train_seed[pred_length:]).reshape(pred_length,1))
    
    metric = mape(predictions, scaler.inverse_transform(test_x))
    logger.info('Predictions complted. Final MAPE is {0}'.format(np.round(metric,2)))
    

