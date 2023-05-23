import keras
from keras import layers

def cnn(input_shape, output_shape):
    model = keras.Sequential()

    model.add(layers.Conv2D(
        16, kernel_size = (2, 2), input_shape = input_shape, 
        padding = "same", activation = "relu")
    )
    model.add(layers.MaxPooling2D(pool_size = 2, strides = (2, 2)))

    model.add(layers.Conv2D(
        8, kernel_size = (2, 2), input_shape = input_shape, 
        padding = "same", activation = "relu")
    )
    model.add(layers.MaxPooling2D(pool_size = 2, strides = (2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(32))
    model.add(layers.Dense(32))
    model.add(layers.Dense(output_shape, activation = "sigmoid"))

    model.compile(
        optimizer = "adam",
        loss = keras.losses.CategoricalCrossentropy(),
        metrics = [keras.metrics.CategoricalAccuracy(name = "Accuracy")]
    )

    return model