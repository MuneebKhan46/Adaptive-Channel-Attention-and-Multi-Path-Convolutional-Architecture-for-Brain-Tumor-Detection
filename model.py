import tensorflow as tf
from tensorflow.keras import layers, models

### ECA (Efficient Channel Attention) Block ###
def eca_block(input_feature, k_size=3, activation='swish'):
    """
    Apply Efficient Channel Attention (ECA) on the input tensor.
    
    Args:
        input_feature: Tensor, input feature map.
        k_size: int, kernel size for 1D convolution.
        activation: str, activation function to apply after Conv1D.
    
    Returns:
        Tensor, feature map after applying ECA.
    """
    channel = input_feature.shape[-1]
    if channel is None:
        raise ValueError("The channel dimension of the input feature must be defined.")

    # Calculate appropriate kernel size for ECA
    k_size = max(int(abs((tf.math.log(tf.cast(channel, tf.float32)) / tf.math.log(2.0) + 1) / 2)), 1)
    k_size = k_size if k_size % 2 else k_size + 1

    # ECA Implementation
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((channel, 1))(avg_pool)
    conv1d = layers.Conv1D(1, kernel_size=k_size, padding='same', use_bias=False)(avg_pool)
    conv1d = layers.Activation(activation)(conv1d)
    conv1d = layers.Reshape((1, 1, channel))(conv1d)

    return layers.Multiply()([input_feature, conv1d])


### Multi-Path Convolutional Blocks ###
def create_block1(filters, kernel_size, strides=1):
    """Single convolution block with MaxPooling and BatchNormalization."""
    return models.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'),
        layers.BatchNormalization()
    ])

def create_block2(filters, kernel_size, strides=1):
    """Double convolution block with Dropout, MaxPooling, and BatchNormalization."""
    return models.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu'),
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25)
    ])

def create_main_block(input_tensor, filters, kernel_size1, kernel_size2):
    """Multi-path block combining different convolutional paths."""
    path1 = create_block1(filters, kernel_size1)(input_tensor)
    path2 = create_block1(filters, kernel_size2)(input_tensor)
    path3 = create_block2(filters, kernel_size2)(input_tensor)
    return layers.Concatenate()([path1, path2, path3])


### Attention Block (CBAM) ###
def attention_block(x, ratio=8):
    """
    Apply Convolutional Block Attention Module (CBAM) on input tensor.
    
    Args:
        x: Tensor, input feature map.
        ratio: int, reduction ratio for channel attention.
    
    Returns:
        Tensor, feature map after applying CBAM.
    """
    channel = x.shape[-1]

    # Channel Attention
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)

    shared_dense_one = layers.Dense(channel // ratio, activation='swish',
                                    kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_out = shared_dense_two(shared_dense_one(avg_pool))
    max_out = shared_dense_two(shared_dense_one(max_pool))

    cbam_feature = layers.Add()([avg_out, max_out])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.Multiply()([x, cbam_feature])


### Model Definition with ECA and Attention ###
def build_model_with_eca(input_shape=(224, 224, 3), num_classes=4, eca_k_size=3, eca_activation='swish'):
    """
    Build the CNN model with ECA and CBAM attention mechanisms.
    
    Args:
        input_shape: tuple, input shape of the model.
        num_classes: int, number of output classes.
        eca_k_size: int, kernel size for ECA block.
        eca_activation: str, activation function for ECA block.
    
    Returns:
        Compiled TensorFlow Keras model.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial Convolutional Block
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # First Multi-Path Block with ECA
    x = create_main_block(x, 32, 1, 3)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = eca_block(x, k_size=eca_k_size, activation=eca_activation)

    # Second Multi-Path Block with ECA
    x = create_main_block(x, 64, 1, 3)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = eca_block(x, k_size=eca_k_size, activation=eca_activation)

    # Third Multi-Path Block with ECA
    x = create_main_block(x, 128, 3, 5)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = eca_block(x, k_size=eca_k_size, activation=eca_activation)

    # Apply CBAM Attention
    x = attention_block(x)

    # Final Classification Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Build and Compile Model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


### Build and Summarize the Model ###
if __name__ == "__main__":
    model = build_model_with_eca()
    model.summary()
