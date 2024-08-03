import tensorflow as tf
import argparse
import onnx

def tflite_model_details(model_path):
    tf.lite.experimental.Analyzer.analyze(model_path)

def onnx_model_details(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    for node in graph.node:
        print(f"Layer name: {node.name}, Type: {node.op_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to a model")
    args = parser.parse_args()

    if args.model_path.endswith(".tflite"):
        tflite_model_details(args.model_path)
    elif args.model_path.endswith(".onnx"):
        onnx_model_details(args.model_path)