import tensorflow as tf
from tensorflow.keras import layers, models

def eca_block(input_feature, k_size=3, activation='swish'):
    """Efficient Channel Attention (ECA) block."""
    channel = input_feature.shape[-1]
    k_size = max(int(abs((tf.math.log(tf.cast(channel, tf.float32)) / tf.math.log(2.0) + 1) / 2)), 1)
    k_size = k_size if k_size % 2 else k_size + 1

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((channel, 1))(avg_pool)
    conv1d = layers.Conv1D(1, kernel_size=k_size, padding='same', use_bias=False)(avg_pool)
    conv1d = layers.Activation(activation)(conv1d)
    conv1d = layers.Reshape((1, 1, channel))(conv1d)

    return layers.Multiply()([input_feature, conv1d])

def create_main_block(input_tensor, filters, kernel_size1, kernel_size2):
    """Multi-path convolutional block."""
    path1 = models.Sequential([
        layers.Conv2D(filters, kernel_size1, padding='same', activation='relu'),
        layers.BatchNormalization()
    ])(input_tensor)

    path2 = models.Sequential([
        layers.Conv2D(filters, kernel_size2, padding='same', activation='relu'),
        layers.BatchNormalization()
    ])(input_tensor)

    path3 = models.Sequential([
        layers.Conv2D(filters, kernel_size2, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2, padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25)
    ])(input_tensor)

    return layers.Concatenate()([path1, path2, path3])

def build_model_with_eca(input_shape=(224, 224, 3), num_classes=4):
    """Build CNN model with ECA and multi-path architecture."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = create_main_block(x, 64, 3, 5)
    x = eca_block(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)
