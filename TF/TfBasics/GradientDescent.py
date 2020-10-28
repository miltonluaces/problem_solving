#region Imports

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

#endregion

# Simple prediction of house prices based on house size
def HousePricePrediction():
    # Generate random house sizes
    nHouses = 160
    np.random.seed(42)
    Sizes = np.random.randint(low=1000, high=3500, size=nHouses ) 

    # Generate house prices from house size + random noise
    np.random.seed(42)
    Prices = Sizes * 100.0 + np.random.randint(low=20000, high=70000, size=nHouses)  

    # Define training/testing set
    nTrain = math.floor(nHouses * 0.7)

    def normalize(array): 
        return (array - array.mean()) / array.std()

    trainDsSize = np.asarray(Sizes[:nTrain])
    trainDsPrice = np.asanyarray(Prices[:nTrain:])
    tranDsSizeNorm = normalize(trainDsSize)
    trainDsPriceNorm = normalize(trainDsPrice)

    testDsSize = np.array(Sizes[nTrain:])
    testDsPrice = np.array(Prices[nTrain:])
    testDsSizeNorm = normalize(testDsSize)
    testDsPriceNorm = normalize(testDsPrice)

    #  Set up placeholders that get updated as we descend down the gradient
    size = tf.placeholder("float", name="size")
    price = tf.placeholder("float", name="price")

    # 1. Define the variables holding the size_factor and price we set during training. Normal random initializing 
    sizeFactor = tf.Variable(np.random.randn(), name="size_factor")
    priceOffset = tf.Variable(np.random.randn(), name="price_offset")

    # 2. Define calculation: predicted price = (size_factor * house_size ) + price_offset
    pred = tf.add(tf.multiply(sizeFactor, size), priceOffset)

    # 3. Define the Loss Function (error meassure) : MSE
    cost = tf.reduce_sum(tf.pow(pred-price, 2))/(2*nTrain)

    # 4. Define optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    # 5. Calculate
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        epochs = 50
        display_every = 2
        
        for i in range(epochs):
            for (x, y) in zip(tranDsSizeNorm, trainDsPriceNorm):
                sess.run(optimizer, feed_dict={size: x, price: y})
            if (i + 1) % display_every == 0:
                c = sess.run(cost, feed_dict={size: tranDsSizeNorm, price:trainDsPriceNorm})
                print("iteration #:", '%04d' % (i + 1), "cost=", "{:.9f}".format(c), "sizeFactor=", sess.run(sizeFactor), "priceOffset=", sess.run(priceOffset))
       
        # 6. Result
        training_cost = sess.run(cost, feed_dict={size: tranDsSizeNorm, price: trainDsPriceNorm})
        print("cost=", training_cost, "sizeFactor=", sess.run(sizeFactor), "priceOffset=", sess.run(priceOffset), '\n')


        # 7. Plot 
        sizeMean = trainDsSize.mean()
        sizeSd = trainDsSize.std()
        priceMean = trainDsPrice.mean()
        priceSd = trainDsPrice.std()

        plt.rcParams["figure.figsize"] = (10,8)
        plt.figure()
        plt.ylabel("Price")
        plt.xlabel("Size (sq.ft)")
        plt.plot(trainDsSize, trainDsPrice, 'go', label='Training data')
        plt.plot(testDsSize, testDsPrice, 'mo', label='Testing data')
        plt.plot(tranDsSizeNorm * sizeSd + sizeMean,(sess.run(sizeFactor) * tranDsSizeNorm + sess.run(priceOffset)) * priceSd + priceMean, label='Learned Regression')
        plt.legend(loc='upper left')
        plt.show()

   
def HousePricePredictionAnim():
    # Generate random house sizes
    nHouses = 160
    np.random.seed(42)
    Sizes = np.random.randint(low=1000, high=3500, size=nHouses ) 

    # Generate house prices from house size + random noise
    np.random.seed(42)
    Prices = Sizes * 100.0 + np.random.randint(low=20000, high=70000, size=nHouses)  

    # Plot generated house and size (bx = blue x)
    plt.plot(Sizes, Prices, "bx")  
    plt.ylabel("Price")
    plt.xlabel("Size")
    plt.show()

    # Normalize values to prevent under/overflows.
    def normalize(array): 
        return (array - array.mean()) / array.std()

    # Define number of training samples, 0.7 = 70%.  (We take the first 70% since values are random)
    nTrain = math.floor(nHouses * 0.7)

    # Define training/testing set
    trainDsSize = np.asarray(Sizes[:nTrain])
    trainDsPrice = np.asanyarray(Prices[:nTrain:])
    tranDsSizeNorm = normalize(trainDsSize)
    trainDsPriceNorm = normalize(trainDsPrice)

    testDsSize = np.array(Sizes[nTrain:])
    testDsPrice = np.array(Prices[nTrain:])
    testDsSizeNorm = normalize(testDsSize)
    testDsPriceNorm = normalize(testDsPrice)

    #  Set up placeholders that get updated as we descend down the gradient
    size = tf.placeholder("float", name="size")
    price = tf.placeholder("float", name="price")

    # 1. Define the variables holding the size_factor and price we set during training. Normal random initializing 
    sizeFactor = tf.Variable(np.random.randn(), name="size_factor")
    priceOffset = tf.Variable(np.random.randn(), name="price_offset")

    # 2. Define calculation: predicted price = (size_factor * house_size ) + price_offset
    pred = tf.add(tf.multiply(sizeFactor, size), priceOffset)

    # 3. Define the Loss Function (error meassure) : MSE
    cost = tf.reduce_sum(tf.pow(pred-price, 2))/(2*nTrain)

    # 4. Define optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    # 5. Calculate
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        epochs = 50
        display_every = 2
        
        # calculate the number of lines to animation
        fit_num_plots = math.floor(epochs/display_every)
        # add storage of factor and offset values from each epoch
        fit_size_factor = np.zeros(fit_num_plots)
        fit_price_offsets = np.zeros(fit_num_plots)
        fit_plot_idx = 0    

       # training
        for i in range(epochs):
            # Fit all training data
            for (x, y) in zip(tranDsSizeNorm, trainDsPriceNorm):
                sess.run(optimizer, feed_dict={size: x, price: y})
            # Display current status
            if (i + 1) % display_every == 0:
                c = sess.run(cost, feed_dict={size: tranDsSizeNorm, price:trainDsPriceNorm})
                print("iteration #:", '%04d' % (i + 1), "cost=", "{:.9f}".format(c), "size_factor=", sess.run(sizeFactor), "price_offset=", sess.run(priceOffset))
                # Save the fit size_factor and price_offset to allow animation of learning process
                fit_size_factor[fit_plot_idx] = sess.run(sizeFactor)
                fit_price_offsets[fit_plot_idx] = sess.run(priceOffset)
                fit_plot_idx = fit_plot_idx + 1

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={size: tranDsSizeNorm, price: trainDsPriceNorm})
        print("Trained cost=", training_cost, "size_factor=", sess.run(sizeFactor), "price_offset=", sess.run(priceOffset), '\n')


       # Plot of training and test data, and learned regression
    
        # get values used to normalized data so we can denormalize data back to its original scale
        sizeMean = trainDsSize.mean()
        sizeSd = trainDsSize.std()

        priceMean = trainDsPrice.mean()
        priceSd = trainDsPrice.std()

        # Plot the graph
        plt.rcParams["figure.figsize"] = (10,8)
        plt.figure()
        plt.ylabel("Price")
        plt.xlabel("Size (sq.ft)")
        plt.plot(trainDsSize, trainDsPrice, 'go', label='Training data')
        plt.plot(testDsSize, testDsPrice, 'mo', label='Testing data')
        plt.plot(tranDsSizeNorm * sizeSd + sizeMean,(sess.run(sizeFactor) * tranDsSizeNorm + sess.run(priceOffset)) * priceSd + priceMean, label='Learned Regression')
        plt.legend(loc='upper left')
        plt.show()

        #Plot another graph that animation of how Gradient Descent sequentually adjusted size_factor and price_offset to find the "best" fit
        fig, ax = plt.subplots()
        line, = ax.plot(house_size, house_price)

        plt.rcParams["figure.figsize"] = (10,8)
        plt.title("Gradient Descent Fitting Regression Line")
        plt.ylabel("Price")
        plt.xlabel("Size (sq.ft)")
        plt.plot(train_house_size, train_price, 'go', label='Training data')
        plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

        def animate(i):
            line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean)  # update the data
            line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offsets[i]) * train_price_std + train_price_mean)  # update the data
            return line,
 
         # Init only required for blitting to give a clean slate.
        def initAnim():
            line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
            return line,

        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim, interval=1000, blit=True)
        plt.show()   


