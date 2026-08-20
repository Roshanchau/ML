import numpy as np;
import tensorflow as tf;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import SimpleRNN, Dense;


# defining the input text and preparing charcter set
text = "I am learning NLP and deep learning. So, lets grind and maintain consistency";
chars = sorted(list(set(text)));
char_to_index = {char: i 
                 for i , char in enumerate(chars)
                 };
index_to_char = {i: char 
                 for i , char in enumerate(chars)
                 };

# creating sequences and labels

seq_length = 3;
sequences = [];
labels = [];

for i in range(len(text) - seq_length):
    seq= text[i: i + seq_length];
    label = text [ i + seq_length];
    sequences.append([char_to_index[char] for char in seq]);
    labels.append(char_to_index[label]);

x= np.array(sequences);
y= np.array(labels);

# converting sequences and labels to One-Hot Encoding(basically converting
# each tokens into an array representation of 0s and 1s so that NN can understand and process it better
# )
X_one_hot = tf.one_hot(x, len(chars));
Y_one_hot = tf.one_hot(y, len(chars));


# building the RNN model

model = Sequential();
model.add(SimpleRNN(50, input_shape= (seq_length, len(chars)),
    activation = 'tanh'));
model.add(Dense(len(chars) , activation= 'softmax'));


# compiling and training the model
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics =['accuracy']);
model.fit(X_one_hot, Y_one_hot , epochs = 100);

start_seq = "I am l"
generated_text = start_seq

for i in range(50):
    x = np.array([[char_to_index[char] for char in generated_text[-seq_length:]]])
    x_one_hot = tf.one_hot(x, len(chars))
    prediction = model.predict(x_one_hot)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    generated_text += next_char

print("Generated Text:")
print(generated_text)