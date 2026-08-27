import tensorflow as tf;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense, Flatten;
from tensorflow.keras.optimizers import Adam;
from tensorflow.keras.losses import SparseCategoricalCrossentropy;
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy;

# load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist;
(x_train, y_train) , (x_test , y_test) = mnist.load_data();
x_train,x_test = x_train /255.0 , x_test/255.0;

# build the model
model = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(128 , activation='relu'),
    Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer = Adam(),
                loss= SparseCategoricalCrossentropy(),
                metrics = [SparseTopKCategoricalAccuracy]
              )


# train the model
model.fit(x_train , y_train , epochs=5);

# evaluate the model
test_loss , test_acc = model.evaluate(x_test , y_test);
print(f'\nTest accuracy: {test_acc}');