import numpy as np
import tensorflow as tf
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices={"resnet_v1_34", "resnet_v1_50", "mobilenet_v2"})
args = parser.parse_args()

tf_model = f'models/tf_{args.model}'
tflite_model_name = f'models/{args.model}.tflite'
checkpoint_dir = f'models/{args.model}'
checkpoint_file = f'{args.model}.ckpt'
BATCH_SIZE = 10

with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph(f'{checkpoint_dir}/{checkpoint_file}.meta')
    saver.restore(sess, f'{checkpoint_dir}/{checkpoint_file}')

    graph = tf.compat.v1.get_default_graph()

    if False:
        for op in graph.get_operations():
            print(op.name)

    if args.model == "resnet_v1_50" or args.model == "mobilenet_v2":
        input_tensor = graph.get_tensor_by_name('input_image:0')
        is_training_tensor = graph.get_tensor_by_name('is_training:0')
        inputs = {'input_image': input_tensor, 'is_training': is_training_tensor}
    elif args.model == "resnet_v1_34":
        input_tensor = graph.get_tensor_by_name('Placeholder:0')
        inputs = {'Placeholder': input_tensor}

    if args.model == "resnet_v1_50" or args.model == "resnet_v1_34":
        output_tensor = graph.get_tensor_by_name(f'{args.model}/predictions/Softmax:0')
    elif args.model == "mobilenet_v2":
        output_tensor = graph.get_tensor_by_name('MobilenetV2/Predictions/Softmax:0')

    input_data = np.random.randn(BATCH_SIZE, 224, 224, 3).astype(np.float32)
    if args.model == "resnet_v1_50" or args.model == "mobilenet_v2":
        start = time.time()
        feed_dict = {input_tensor: input_data, is_training_tensor: False}
        output_data = sess.run(output_tensor, feed_dict=feed_dict)
        end = time.time()
    elif args.model == "resnet_v1_34":
        start = time.time()
        for i in range(BATCH_SIZE):
            feed_dict = {input_tensor: input_data[i:i+1]}
            output_data = sess.run(output_tensor, feed_dict=feed_dict)
        end = time.time()

    if not tf.io.gfile.exists(tf_model):
        tf.compat.v1.saved_model.simple_save(
            sess,
            tf_model,
            inputs=inputs,
            outputs={'output': output_tensor}
        )

    print(f"Average TF inference time: {(end - start) / BATCH_SIZE} seconds, FPS: {BATCH_SIZE / (end - start)}")

if not tf.io.gfile.exists(tflite_model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the converted model
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=tflite_model_name)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = np.ones((1, 224, 224, 3), dtype=np.float32)
start = time.time()
for _ in range(BATCH_SIZE):
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
end = time.time()
print(f"Average TFLite inference time: {(end - start) / BATCH_SIZE} seconds, FPS: {BATCH_SIZE / (end - start)}")

