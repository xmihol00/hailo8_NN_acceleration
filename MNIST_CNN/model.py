import tensorflow as tf

def build_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="input")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(name="maxpool1")(x)
    x = tf.keras.layers.Conv2D(16, 3, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D(name="maxpool2")(x)
    x = tf.keras.layers.Conv2D(16, 3, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(10, name="dense1")(x)
    outputs = tf.keras.layers.Softmax(name="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="MNIST_CNN")
    return model

def build_conv_only_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="input")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(name="maxpool1")(x)
    x = tf.keras.layers.Conv2D(16, 3, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D(name="maxpool2")(x)
    x = tf.keras.layers.Conv2D(16, 3, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    outputs = tf.keras.layers.ReLU(name="relu3")(x)
    model = tf.keras.Model(inputs, outputs, name="MNIST_CONV_ONLY")
    return model