import numpy as np
import tensorflow as tf
import time

checkpoint_dir = 'models/resnet_v1_50'
checkpoint_file = 'resnet_v1_50.ckpt'
BATCH_SIZE = 1

if not tf.io.gfile.exists('models/tf_resnet_v1_50'):
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(f'{checkpoint_dir}/{checkpoint_file}.meta')
        saver.restore(sess, f'{checkpoint_dir}/{checkpoint_file}')

        graph = tf.compat.v1.get_default_graph()

        input_tensor = graph.get_tensor_by_name('input_image:0')
        is_training_tensor = graph.get_tensor_by_name('is_training:0')
        output_tensor = graph.get_tensor_by_name('resnet_v1_50/predictions/Softmax:0')

        start = time.time()
        input_data = np.random.randn(BATCH_SIZE, 224, 224, 3).astype(np.float32)
        feed_dict = {input_tensor: input_data, is_training_tensor: False}
        output_data = sess.run(output_tensor, feed_dict=feed_dict)
        end = time.time()

        if not tf.io.gfile.exists('models/tf_resnet_v1_50'):
            tf.compat.v1.saved_model.simple_save(
                sess,
                'models/tf_resnet_v1_50',
                inputs={'input_image': input_tensor, 'is_training': is_training_tensor},
                outputs={'output': output_tensor}
            )

        print(output_data.flatten()[0], f"Average inference time: {(end - start) / BATCH_SIZE} seconds")

if not tf.io.gfile.exists('models/resnet_v1_50.tflite'):
    converter = tf.lite.TFLiteConverter.from_saved_model('models/tf_resnet_v1_50')
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the converted model
    with open('models/resnet_v1_50.tflite', 'wb') as f:
        f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='models/resnet_v1_50.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = np.ones((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(output.flatten()[0])
