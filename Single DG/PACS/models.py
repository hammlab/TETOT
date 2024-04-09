import tensorflow as tf

def representationNN(rep_dim, output_dim):
    
    inputs = tf.keras.layers.Input(shape=(rep_dim), name='inputs')
    
    x = inputs
    
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(output_dim)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def classificationNN(rep_dim, num_classes):
    inputs = tf.keras.layers.Input(shape=(rep_dim), name='inputs')
    
    x = inputs
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

