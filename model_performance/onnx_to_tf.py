import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

def convert_onnx_to_tf(onnx_model_path):
    model = onnx.load(onnx_model_path)
    tf_rep = prepare(model, training_mode=True)
    
    return tf_rep

def run_inference(tf_rep, input_data):
    print(tf_rep.outputs, tf_rep.tensor_dict)
    input_tensor_name = tf_rep.inputs[0]
    output_tensor_name = tf_rep.outputs[0]
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(tf_rep.graph.as_graph_def(), name="")
        
        with tf.compat.v1.Session(graph=graph) as session:
            input_tensor = graph.get_tensor_by_name(input_tensor_name + ":0")
            output_tensor = graph.get_tensor_by_name("add_20:0")
            
            print("Inferring...")
            output_data = session.run(output_tensor, feed_dict={input_tensor: input_data})
            print("Inference complete!")
    
    return output_data

onnx_model_path = "models/resnet_v1_18.onnx"
tf_rep = convert_onnx_to_tf(onnx_model_path)

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

output_data = run_inference(tf_rep, input_data)
print("Inference output:", output_data)
