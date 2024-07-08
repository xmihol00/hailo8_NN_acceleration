import tensorflow as tf
import numpy as np
import idx2numpy as idx

def build_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="img")
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

model = build_model()
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

X_train = idx.convert_from_file("../../mnist_ML/mnist/train-images.idx3-ubyte") / 255  # download the dataset from http://yann.lecun.com/exdb/mnist/
X_train = X_train.reshape(-1, 28, 28, 1)
y_train = idx.convert_from_file("../../mnist_ML/mnist/train-labels.idx1-ubyte")

model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

tf.saved_model.save(model, "weights")

CREATE_TFLITE = False
if CREATE_TFLITE:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS 
    ]
    tflite_model = converter.convert()  # may cause warnings in jupyter notebook, don"t worry.
    tflite_model_path = "model.tflite"
    with tf.io.gfile.GFile(tflite_model_path, "wb") as f:
        f.write(tflite_model)

