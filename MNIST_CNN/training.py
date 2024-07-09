import tensorflow as tf
import idx2numpy as idx
from model import build_model

if __name__ == "__main__":
    model = build_model()
    model.summary()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    X_train = idx.convert_from_file("../../mnist_ML/mnist/train-images.idx3-ubyte")  # download the dataset from http://yann.lecun.com/exdb/mnist/
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
