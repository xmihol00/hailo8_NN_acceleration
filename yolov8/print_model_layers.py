import tensorflow as tf

saved_model_dir = "models/graph_frozen.pb"
model = tf.saved_model.load(saved_model_dir)

for layer in model.layers:
    print(layer.name)

