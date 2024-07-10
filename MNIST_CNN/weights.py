import tensorflow as tf
from model import build_model, build_conv_only_model
import numpy as np

def weights_to_C_array(weights, name):
    weights_str = "{\n"
    if len(weights.shape) == 2:
        for row in weights:
            weights_str += "\t" + ", ".join([f"{w: .7f}" for w in row]) + ",\n"
        
        weights_str = weights_str[:-2] + "\n"
    else:
        weights_str += "\t" + ", ".join([f"{w: .7f}" for w in weights]) + "\n"

    weights_str += "}"

    return f"const float {name}[{len(weights.flatten())}] = {weights_str};"

if __name__ == "__main__":
    model = build_model()
    conv_only_model = build_conv_only_model()
    model.load_weights("weights/variables/variables")

    for layer, conv_only_layer in zip(model.layers, conv_only_model.layers):
        conv_only_layer.set_weights(layer.get_weights())

    for layer in model.layers:
        if "dense" in layer.name:
            dense_weights, dense_biases = layer.get_weights()
    
    print(f"Shapes of dense weights and biases: {dense_weights.shape}, {dense_biases.shape}")

    ones = np.ones((1, 32, 32, 1))
    prediction = model.predict(ones)
    conv_only_prediction = conv_only_model.predict(ones)
    prediction_np = np.matmul(conv_only_prediction.flatten(), dense_weights) + dense_biases
    prediction_np = tf.nn.softmax(prediction_np).numpy()

    prediction_python = np.zeros((1, 10))
    conv_only_prediction = conv_only_prediction.flatten()    
    for i, row in enumerate(dense_weights):
        prediction_python += row * conv_only_prediction[i]
    prediction_python += dense_biases
    prediction_python = tf.nn.softmax(prediction_python).numpy()

    print("Result is the same: ", np.isclose(prediction, prediction_np).all())
    print("Result is the same: ", np.isclose(prediction, prediction_python).all())
    with open("weights.h", "w") as f:
        f.write(weights_to_C_array(dense_weights, "weights"))
        f.write("\n\n")
        f.write(weights_to_C_array(dense_biases, "biases"))
    
