import tensorflow as tf
from utils import paths

model = tf.keras.models.load_model(paths.models_path / "21-50 04_03-21")

for layer in model.layers:
    print(layer)

print()
