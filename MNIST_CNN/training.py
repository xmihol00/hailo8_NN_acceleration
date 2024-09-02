import tensorflow as tf
from model import build_model
import numpy as np
import tensorflow_model_optimization as tfmot


if __name__ == "__main__":
    model = build_model()
    model = tfmot.quantization.keras.quantize_model(model)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    with open("mnist_train_X.bin", "rb") as f:
        X_train = f.read()
        X_train = np.frombuffer(X_train, dtype=np.uint8).reshape(-1, 32, 32, 1)
    with open("mnist_train_y.bin", "rb") as f:
        y_train = f.read()
        y_train = np.frombuffer(y_train, dtype=np.uint8)
    
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)
    tf.saved_model.save(model, "weights")
    model.summary()
    for layer in model.layers:
        print(layer.name, layer.input_shape, layer.output_shape)

    CREATE_TFLITE = False
    if CREATE_TFLITE:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS 
        ]
        tflite_model = converter.convert()
        tflite_model_path = "model.tflite"
        with tf.io.gfile.GFile(tflite_model_path, "wb") as f:
            f.write(tflite_model)
