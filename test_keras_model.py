import tensorflow as tf
import keras

new_model = tf.keras.models.load_model('keras_model_test.model')

predictions = new_model.predict([[[153,256,99,0,86,65]]])

print(predictions)
