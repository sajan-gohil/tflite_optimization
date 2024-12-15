import traceback
import tensorflow as tf


def quantize_model(input_path, output_path):
    model = tf.saved_model.load(input_path)
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([None, 256, 256, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # Apply Dynamic range quantization (float to int8, without requirement of representative dataset)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Built-in operations
        tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow Select Ops
    ]
    converter.allow_custom_ops = True
    quantized_model = converter.convert()

    # Save the quantized model
    with open(output_path, 'wb') as f:
        f.write(quantized_model)

    print("Quantization complete. Quantized model saved at ", output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default=None,
                        help="Path to tensorflow model")
    parser.add_argument("--output_path",
                        type=str,
                        default=None,
                        help="Path where quantized model will be saved")
    args = parser.parse_args()
    assert args.input_path and args.output_path, "BOTH --input_path AND --output_path ARE REQUIRED"

    quantize_model(args.input_path, args.output_path)
