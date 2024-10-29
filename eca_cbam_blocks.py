from tensorflow.keras import layers

def eca_block(input_feature, k_size=3):
    channels = input_feature.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    conv1d = layers.Conv1D(1, kernel_size=k_size, padding='same', use_bias=False)(avg_pool[:, :, None])
    return layers.Multiply()([input_feature, conv1d])

def attention_block(x):
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    fc = layers.Dense(x.shape[-1] // 8, activation='swish')(avg_pool + max_pool)
    return layers.Multiply()([x, fc])
