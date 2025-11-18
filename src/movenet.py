import tensorflow as tf

path = 'models/movenet/movenet_thunder.tflite'
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()