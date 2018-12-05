# Introduction to Keras

  - Deep learning framework written in Python
  - Open source, created in 2014.
  - Popular & active community
  
  - resources built on top of keras: github.com/fchollet/keras-resource
  - Examples: github.com/fchollet/keras/tree/master/examples
  - Models: github.com/fchollet/deep-learning-models
  - Data: github.com/fchollet/keras/tree/master/keras/datasets
  
  - Choice of backends available:
    - Tensorflow
    - theano
    - CNTK    
  - Swap backends easily, run on CPUs or GPUs
  
  - A minimal Keras example:
  
    - from keras.models import Sequential
    - from keras.layers import Dense
    - from keras.optimizers import SGD
    
    - #load data
    - from keras.datasets import mnist
    - (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    - #Define model
    - model = Sequential()
    
    - #Add layers
    - model.add(Dense(256, activation='sigmoid', input_shape=(784,)))
    
    - #Compile model with loss and optimizer
    - model.compile(loss='catgorical_crossentropy', optimizer=SGD(), metric=['accuracy'])
    
    - #Train network
    - model.fit(x_train, y_train, batch_size=128, apochs=10, validation_data=(x_test, y_test))
    
  - Install Keras
  
    - #Install tensorflow backend
    - pip install tensorflow

    - #Optionally install other dependencies
    - pip install h5py graphviz pydot

    - #Install Keras
    - pip install keras
  
  
# Sequential Models:

  - Sequential:
    - simply stack layers sequentially
  - Non-sequential models: Model
  - More general models, uses a functional API
  
  - Keras layers:
    - core abstraction for every mdel
    - sequential models layers have input, output, input_shape, output_shape
    - Get weights: layer.get_weights()
    - Set weights: layer.set_weights(weights)
    - layer config: layer.get_config()
    
  - Build a sequential model
    - Instantiate a Sequential model
    - Add layers to it one by one using add
    - Compile the model with a loss function, an optimizer and optional evaluation metric ( accuracy )
    - use the data to fit the model
    - evaluate model, persist or deploy model, start new experiment etc.
  
  - Loss functions:
  
    - #Option 1: Importing from loss module (preferred)
    - from keras.losses import mean_squared_error
    - model.compile(loss=mean_squared_error, optimizer=...)
    
    - #Option 2: Using strings
    - model.compile(loss='mean_squared_error', optimizer=...) <---- error prone because of typo
    
  - Optimizers:
  
    - #Option 1: load optimizers from moddule ( preferred )
    - from keras.optimizers import SGD
    
    - sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    - model.compile(loss=..., optimizer=sgd)
    
    - #Option 2: pass string (default parameters will be used)
    - model.compile(loss=..., optimizer='sgd')
    
  - Fit, evaluate and predict
  
    - model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
    
    - #Evaluate on test data
    - evaluate(x_test, y_test, batch_size=32)
    
    - #predict labels
    - predict(x_test, batch_size=32)
    
    
  
  
