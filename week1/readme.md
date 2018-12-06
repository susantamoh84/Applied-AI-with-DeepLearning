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
  
    ```from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD
    
    #load data
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    #Define model
    model = Sequential()
    
    #Add layers
    model.add(Dense(256, activation='sigmoid', input_shape=(784,)))
    
    #Compile model with loss and optimizer
    model.compile(loss='catgorical_crossentropy', optimizer=SGD(), metric=['accuracy'])
    
    #Train network
    model.fit(x_train, y_train, batch_size=128, apochs=10, validation_data=(x_test, y_test))```
    
  - Install Keras
  
    ```#Install tensorflow backend
    pip install tensorflow

    #Optionally install other dependencies
    pip install h5py graphviz pydot

    #Install Keras
    pip install keras```
  
  
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
  
    ``` 
    #Option 1: Importing from loss module (preferred)
    from keras.losses import mean_squared_error
    model.compile(loss=mean_squared_error, optimizer=...)
    
    #Option 2: Using strings
    model.compile(loss='mean_squared_error', optimizer=...) <---- error prone because of typo```
    
  - Optimizers:
  
    ``` 
    #Option 1: load optimizers from moddule ( preferred )
    from keras.optimizers import SGD
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss=..., optimizer=sgd)
    
    #Option 2: pass string (default parameters will be used)
    model.compile(loss=..., optimizer='sgd')```
    
  - Fit, evaluate and predict
  
    ```model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
    
    #Evaluate on test data
    evaluate(x_test, y_test, batch_size=32)
    
    #predict labels
    predict(x_test, batch_size=32)```
       
# Feed Forward Networks

  - Building blocks for MLPs
  
    - Dense layers with activations
    - Use Dropout for regularization
    - Build a Sequential model from Dense and Dropout layers
    
  - Dense layers:
    ``` from keras.layers import Dense
    
    Dense(units,            # Number of output neurons
          activation=None,  # Activation function by name
          use_bias=True,    # Use bias term or not
          kernel_initializer='glorot_uniform', 
          bias_initializer='zeros')```      

  - Dropout Layers:
    ```from keras.datasets import mnist
    from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Dense,Dropout
    
    batch_size = 128
    num_classes = 10
    epochs = 20
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    #Data preprocessing
    
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    #Defining and compiling your model
    
    model = Sequential()
    model.add(Dense(512, activation='relu', 
              input_shape=(784,))) # First layer only
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy])
    
    #Running and evaluating your model
    
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    #above model 98% accuracy ```
    
  - Recurrent neural networks
  
    - Available RNNs in keras
      - SimpleRNN - basic RNN
      - GRU - Gated Recursive UNit (2014)
      - LSTM - Long short-term memory (1997)
      
    - LSTM layers
    
      ```from keras.layers.recurrent import LSTM
      
      LSTM(units,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            recurrent_initializer='orthogonal',
            recurrent_regularizer=Nope,
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=False)               #If return_sequences=True, then return all the values in matrix

      # Embedding layers for first layer only
      # Transform integers into vectors of same length: Example: [3, 12] embedded as [[0.1, 0.5], [1.3 4.2]]
      # Embed a vocabulary into a vector space, apply to sentences; 2-D input mapped to 3-D output, connects to LSTMs
      
      from keras.layers.embeddings import Embedding
      
      Embedding(input_dim,                        # Vocabulary size
                output_dim,                       # Ouput vector length
                embeddings_initializer='uniform',
                mask_zero=False)                  # Mask zero values```
                
    - Sentiment classification for movie reviews
    
      - 25,000 movie reviews from IMDB, labelled good or bad
      - Data available from keras.datasets module
      - Data is pre-processed as sequences of integers
      - Task: classify sentiment from review content
      - Strategy: embed sentences, then learn structure with LSTM
      
      ```# Loading IMDB sentiment data
      
      from keras.preprocessing import sequence
      from keras.models import Sequential
      from keras.layers import Dense, Embedding
      from keras.layers import LSTM
      from keras.datasets import imdb
      
      max_features = 20000 #20000 most comment items
      maxlen = 80          # sequences of length 80
      
      (x_trian, y_train), (x_test, y_test) = \
          imdb.load_data(num_words=max_features)
          
      x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
      x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
      
      model = Sequential()
      model.add(Embedding(max_features, 128))
      model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
      model.add(Dense(1, activation='sigmoid'))
      
      #Run and evaluate model
      model.compile(loss='binary_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])
      
      model.fit(x_train, y_train,
                batch_size=32, epochs=15,
                validation_data=(x_test, y_test))
      
      model.evaluate(x_test, y_test, batch_size=32)```
      
# Beyond sequential models: the functional API

  - non-sequential models: Model
  - Model can be trained and evaluated exactly like Sequential
  - functional apis for Model starts with input(s)
  - we then define output(s) by transforming input(s) iteratively
  
  ``` #Using the functional API
  from keras.layers import Input, Dense
  from keras.models import Model
  
  num_classes = 10
  inputs = Input(shape=(784,))
  
  x = Dense(512, activation='relu')(inputs)
  x = Dense(512, activation='relu')(x)
  predictions = Dense(num_classes, activation='softmax')(x)
  
  #Defining and running a Model
  model = Model(inputs=inputs, outputs=predictions)
  model.compile(optimizer='sgd',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
                
  model.fit(...) # same as before
  ```

# Serializing keras models

  - keras models can be saved and loaded
  - full model: architecture, weights and training configurations (HDF5)
  - Architecture only (JSON or YAML)
  - Weights only (HDF5)
  
  ```#Persisting architecture or weights
  
  from keras.models import model_from_json
  
  # Save model as JSON and weights as HDF5
  json_string = model.to_json() # model.to_yaml()
  model.save_weights('weights.h5')
  
  # Load from JSON and set weights
  model = model_from_json(json_string
  model.load_weights('weights.h5')
  
  # Persisting the full model
  from keras.models import load_model
  model.save('full_model.h5')
  model = load_model('full_model.h5')
  ```
  
# Apache SystemML

  - Provides a R-like language (DML) for data scientists to implement machine learning algorithms
    - all notatons & conventions like R    
  - Analyzes the statements and find the hirearchial execution plan for the statements
  - creates a DAG for the execution of statements.  
  - runs in embeddable, standaone, cluster (hybrid) mode  
  - API in java, scala, python
  
# Invoking SystemML

  - location of script: -f 
  - -nargs be passed via -nvargs
  - -exec option hybrid-spark or hybrid
  
  - Case: In-memory single node
    
    ``` # Java Command Line:
    
    $ java -cp SystemML.jar org.apache.sysml.api.DMLScript \
    -f LinearRegCG.dml -exec singlenode -nvargs X=X.mtx Y=Y.mtx B=B.mtx
    
    $ java -cp systemml-1.0.0-standalone.jar \
    org.apache.sysml.api.DMLScript \
    -f LinearRegCG.dml -nvargs X=X.mtx Y=Y.mtx B=B.mtx
    ```
  - Case: In Hadoop
    
    ``` # Hadoop Command Line:
    $ hadoop jar SystemML.jar \
    -f LinearRegCG.dml -exec hybrid -nvargs X=X.mtx Y=Y.mtx B=B.mtx
    ```    
  - Case: In Spark
    
    ``` # Spark Command Line:
    $ spark-submit --master yarn-client SystemML.jar \
    -f LinearRegCG.dml -nvargs X=X.mtx Y=Y.mtx B=B.mtx
    ```
    
  - Case: RDD/DataFrame
    
    ```$ spark-shell --jars SystemML.jar --driver-memory 3g
    
    import org.apache.sysml.api.mlcontext._
    import org.apache.sysml.api.mlcontext.ScriptFactory._
    val ml = new MLContext(sc)
    val X = // ... RDD, DataFrame, etc.
    val script = dmlFromFile("LinearRegCG.dml").in("X", X)
                  .in(...).out("B")
    val b = ml.execute(script).getDataFrame("B")
    ```
    
  - Case: Python MLContext API.
    
    ```$ pip install systemml
    $ pip show systemml
    $ pyspark --driver-memory 3g
    
    from systemml import MLContext, dmlFromFile
    ml = MLContext(sc)
    X = // ... RDD, DataFrame, NumPy, SciPy etc
    script = dmlFromFile("LinearRegCG.dml").input("X", X)
              .input(...).output("B")
    b = ml.execute(script).getDataFrame("B")
    ```
    
  - Case: MLLearn API
    
    ```$ pip install systemml
    $ pyspark --driver-memory 3g
    
    from systemml import LinearRegression
    
    train_df = // ... RDD, DF
    regr = LinearRegression(spark)
    regr.fit(train_df)              <---- sci-kit learn like apis
    b = regr.predict(test_df)       <---- sci-kit learn like apis
    ```
    
  - Case: Experimental APIs: Keras2DML, Caffe2DML
    
    ```$ pip install systemml
    $ pyspark --driver-memory 3g
    
    from systemml import Keras2DML
    
    train_df = // ... NumPy, SciPy
    sysml_model = Keras2DML(spark, keras_model,...)
    sysml_model.fit(train_df)              <---- sci-kit learn like apis
    b = sysml_model.predict(test_df)       <---- sci-kit learn like apis
    ```
    
    ```$ pip install systemml
    $ pyspark --driver-memory 3g
    
    from systemml import Caffe2DML
    
    train_df = // ... RDD, DataFrame, Numpy, SciPy
    sysml_model = Caffe2DML(spark, 'solver.proto',...)
    sysml_model.fit(train_df)              <---- sci-kit learn like apis
    b = sysml_model.predict(test_df)       <---- sci-kit learn like apis
    ```      
      
# Demo - How to use Apache SystemML on IBM DSX
  
  - SystemML 1.1 requires Spark 2.1 & above
  - Example 1:
      ``` from systemml import MLContext, dml

      ml = MLContext(sc)  
      print(ml.info()

      # Create a simple DML script 'hello word'

      script = dml("""
      print('Hello World');
      """)
      ml.execute(script)

      # Modify script to return a string
      script = dml("""
      s = 'Hello World'
      """).output("s")

      hello_world_str = ml.execute(script).get("s")

      print(hello_world_str)
      ```
  - Example 2:
  
      ```import sys, os
      import matplotlib.pyplot as plt
      import numpy as np
      from sklearn import datasets
      plt.switch_backend('agg')
      
      # dml script for matrix multiplication
      script = """
          X = rand(rows=$nr, cols=1000, sparsity=0.5)
          A = t(X) %*% X
          s = sum(A)
      """
      prog = dml(script).input('$nr', 1e6).output('s') <---- input method to pass parameters to dml
      s = ml.execute(prod).get('s')
      print s
      ```
      
  - Example 3:
  
      ```%matplotlib inline
      diabetes = datasets.load_diabetes()
      diabetes_X = diabetes.data[:, np.newaix, 2]
      diabetes_X_train = diabetes_X[:-20]
      diabetes_X_test = diabetes_X[-20:]
      diabetes_y_train = np.matrix(diabetes.target[:-20]).T
      diabetes_y_test = np.matrix(diabetes.target[-20:]).T
      
      plt.scatter(diabetes_X_train, diabetes_y_train, color="black") <---- scatter plot
      plt.scatter(diabetes_y_test, diabetes_y_test, color="red")
      
      #linear regressopm dml
      script = """
          # add constant feature to X to model intercept
          ones = matrix(1, rows=nrow(X), cols=1)
          X = cbind(X, ones)
          A = t(X) %*% X
          b = t(X) %*% y
          w = solve(A, b)
          bias = as.scalar(w[nrow(w),1])
          w = w[1:nrow(w)-1,]
      """
      
      prog = dml(script).input(X=diabetes_X_train, y=diabetes_y_train).output("w", "bias")
      w, bias = ml.execute(prog).get('w', 'bias')
      w =  w.toNumPy()
      
      plt.scatter(diabetes_X_train, diabetes_y_train, color="black") <---- scatter plot
      plt.scatter(diabetes_X_test, diabetes_y_test, color="red")
      
      plt.scatter(diabetes_X_test, (w*diabetes_X_test)+bias, color="blue", linestyle='dotted') <---- scatter plot
      ```
      
  - Example 4: Linear Regression with batch Gradient Descent
  
      ```%matplotlib inline
      #linear regression gradient descent dml
      script = """
          # add constant feature to X to model intercept
          ones = matrix(1, rows=nrow(X), cols=1)
          X = cbind(X, ones)
          max_iter = 100
          w = matrix(0, rows=ncol(X), cols=1)
          for(i in 1:max_iter){
            XtX = t(X) %*% x
            dw = XtX %*% w - t(X) %*% y
            alpha = (t(dw) %*% dw) / (t(dw) %*% XtX %*% dw)
            w = w - dw*alpha
          }
          bias = as.scalar(w[nrow(w),1])
          w = w[1:nrow(w)-1,]
      """
      
      prog = dml(script).input(X=diabetes_X_train, y=diabetes_y_train).output("w", "bias")
      w, bias = ml.execute(prog).get('w', 'bias')
      w =  w.toNumPy()
      
      plt.scatter(diabetes_X_train, diabetes_y_train, color="black") <---- scatter plot
      plt.scatter(diabetes_X_test, diabetes_y_test, color="red")
      
      plt.scatter(diabetes_X_test, (w*diabetes_X_test)+bias, color="blue", linestyle='dotted') <---- scatter plot
      ```
      
  - Example 5: Conjugate Grident Method
  
      ```%matplotlib inline
      #linear regression gradient descent dml
      script = """
          # add constant feature to X to model intercept
          ones = matrix(1, rows=nrow(X), cols=1)
          X = cbind(X, ones)
          m = ncol(X); i = 1
          max_iter = 20
          w = matrix(0, rows=ncol(X), cols=1)
          dw = - t(X) %*% y; p = - dw
          norm_r2 = sum (dw ^ 2)
          for(i in 1:max_iter){
            q = t(X) %*% ( X %*% p )
            alpha = norm_r2 / sum( p * q)
            w = w + alpha * P
            dw = dw + alpha * q
            old_norm_r2 = norm_r2; norm_r2 = sum(dw ^ 2);
            p = -d2 + (norm_r2 / old_norm_r2) * p
            i = i + 1;
          }
          bias = as.scalar(w[nrow(w),1])
          w = w[1:nrow(w)-1,]
      """
      
      prog = dml(script).input(X=diabetes_X_train, y=diabetes_y_train).output("w", "bias")
      w, bias = ml.execute(prog).get('w', 'bias')
      w =  w.toNumPy()
      
      plt.scatter(diabetes_X_train, diabetes_y_train, color="black") <---- scatter plot
      plt.scatter(diabetes_X_test, diabetes_y_test, color="red")

      plt.scatter(diabetes_X_test, (w*diabetes_X_test)+bias, color="blue", linestyle='dotted') <---- scatter plot
      ```
  - Example 5: Invoke existing dml script     
      
      ```
      from systemml import dmlFromResource
      
      prog = dmlFromResource('scripts/algorithms/LinearRegDS.dml')
                .input(X=diabetes_X_train, y=diabetes_y_train)
                .input('$icpt', 1.0).output("beta_out")
      w = ml.execute(prog).get('beta_out')
      w =  w.toNumPy()
      
      plt.scatter(diabetes_X_train, diabetes_y_train, color="black") <---- scatter plot
      plt.scatter(diabetes_X_test, diabetes_y_test, color="red")

      plt.scatter(diabetes_X_test, (w*diabetes_X_test)+bias, color="blue", linestyle='dotted') <---- scatter plot
      ```

  - Example 6: Invoke existing dml script using scitkit-learn
      
      ```
      from pyspark.sql import SQLContext
      from systemml.mllearn import LinearRegression
      sqlCtx = SQLContext(sc)
      
      regr = LinearRegression(sqlCtx)
      regr.fit(diabetes_X_train, diabetes_y_train)
      predictions = regr.predict(
      
      plt.scatter(diabetes_X_train, diabetes_y_train, color="black") <---- scatter plot
      plt.scatter(diabetes_X_test, diabetes_y_test, color="red")

      plt.scatter(diabetes_X_test, (w*diabetes_X_test)+bias, color="blue", linestyle='dotted') <---- scatter plot
      ```    
    
  - Example 7: Invoke keras model with systemml ( Requirement install OpenBLAS )
      
      ```
      from mlxtend.data import mnist_data
      import numpy as np
      from sklearn.utils import shuffle
      
      #Download the MNIST dataset
      X, y = mnist_data()
      X, y = shuffle(X, y)
      
      #Split the data into training and test
      n_samples = len(X)
      X_train = X[:int(0.9 * n_samples)]
      y_train = y[:int(0.9 * n_samples)]
      X_test = X[int(0.9 * n_samples):]
      y_test = y[int(0.9 * n_samples):]
      
      #import keras
      from keras.models import Sequential
      from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
      from keras import backend as K
      from keras.models import Model
      
      #Create the network
      input_shape = (1,28,28) if K.image_data_format() == 'channels_first else (28,28,1))
      keras_model = Sequential()
      keras_model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding="same"))
      keras_model.add(MaxPooling2D(pool_size=(2, 2)))
      keras_model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding="same"))
      keras_model.add(MaxPooling2D(pool_size=(2, 2)))
      keras_model.add(Flatten())
      keras_model.add(Dense(512, activation='relu'))
      keras_model.add(Dropout(0.5))
      keras_model.add(Dense(10, activation='softmax'))
      
      #Scale the input features
      scale = 0.00390625
      X_train = X_train * scale
      X_test = X_test * scale
      
      #Train the model
      from systemml.mllearn import Keras2DML
      sysml_model = Keras2DML(spark, keras_model, input_shape=(1,28,28), weights='weights_dir')
      sysml_model.setConfigProperty('sysml.native.bias', 'openbias')
      sysml_model.setConfigProperty('sysml.native.bias.directory', os.path.join(os.getcwd(), 'OpenBLAS-0.2.20/'))
      # sysml_model.setGPU(True).setForceGPU(True)
      sysml_model.summary()
      sysml_model.fit(X_train, y_train)
      ```

# Introduction to DeepLearning4j

