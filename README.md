# TFlite Optimization
Deep learning model optimization demo using tensorflow lite


# Setup
Make sure you have Python 3.10+ installed.
Install Poetry: `pip install poetry`
Clone this repository and navigate to the project directory.
Run `poetry install` to install the dependencies.


Alternatively, run `pip install -r requirements.txt`


# Usage

Current configurations are for (movenet)[https://www.kaggle.com/models/google/movenet/], a pose landmark estimation model. Some changes might be required for other models.

## Model quantization

`model_optimization/quantize_model.py`: quantie a tensorflow model in from SavedModel format to tflite format. Dynamic range quantization (float to int8) is used, covered by `tf.lite.Optimize.DEFAULT`. This optimization is suitable for cpu/edge devices and does not require any representative dataset.

Example:
```
 python3 model_optimization/quantize_model.py --input_path ./models/movenet-tensorflow2-multipose-lightning-v1/ --output_path ./models/optimized_movenet.tflite
```


## Model inference

`model_optimization/infer.py`: use a tflite model to infer pose landmarks from an input image. The script displays a visualization and saves it as output.jpg.

Example:
```
python3 model_optimization/infer.py --model_path ./optimized_movenet.tflite --input_image /home/srg/Downloads/jumping_jack.jpg
```