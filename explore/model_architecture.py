import tensorflow as tf
import paths

model = tf.keras.models.load_model(paths.models_path / "19_32---14_02-21.mdl")

for layer in model.layers:
    print(layer.units)

print()
