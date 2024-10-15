
import osimport time
import cv2
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Concatenate,GlobalAveragePooling2D


result_file_path = "/Untitled Folder/Results.csv"
dataset_dir = "/Untitled Folder/Dataset Folder/BTD/Training"
test_dir = "/Untitled Folder/Dataset Folder/BTD/Testing"
classes = ["glioma", "meningioma", "pituitary", "notumor"]


def load_data(dataset_dir, classes, width, height):
    images = []
    image_labels = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                try:
                    im = Image.open(image_path).convert('RGB')
                    im = im.resize((width, height))
                    im = np.array(im)
                    images.append(im)
                    image_labels.append(i)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

    return np.array(images), np.array(image_labels)


def eca_block(input_feature, k_size=3, activation='sigmoid'):
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    
    avg_pool = layers.Reshape((input_feature.shape[-1], 1))(avg_pool)
    
    conv1d = layers.Conv1D( 1, kernel_size=k_size, padding='same', use_bias=False, activation=activation)(avg_pool)
    conv1d = layers.Reshape((1, 1, input_feature.shape[-1]))(conv1d)
    scale = layers.Multiply()([input_feature, conv1d])
    return scale


def build_model_with_eca(eca_activation='sigmoid'):
    input_layer = Input(shape=(128, 128, 3))

    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = eca_block(x, activation=eca_activation)
    x = layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    conv_path1 = Conv2D(64, (1, 1), padding='same')(x)
    conv_path1 = eca_block(conv_path1, activation=eca_activation)  # Add ECA block here
    conv_path1 = layers.Activation('relu')(conv_path1)

    conv_path2 = Conv2D(64, (3, 3), padding='same')(x)
    conv_path2 = eca_block(conv_path2, activation=eca_activation)  # Add ECA block here
    conv_path2 = layers.Activation('relu')(conv_path2)

    maxpool_path = MaxPooling2D((2, 2), strides=1, padding='same')(x)
    dropout_path = Dropout(0.25)(maxpool_path)

    block1_output = Concatenate()([conv_path1, conv_path2, dropout_path])

    # Block 2 - Three parallel paths (with 128 filters)
    # Path 1
    conv_path1_b2 = Conv2D(128, (1, 1), padding='same')(block1_output)
    conv_path1_b2 = eca_block(conv_path1_b2, activation=eca_activation)  # Add ECA block here
    conv_path1_b2 = layers.Activation('relu')(conv_path1_b2)

    # Path 2
    conv_path2_b2 = Conv2D(128, (3, 3), padding='same')(block1_output)
    conv_path2_b2 = eca_block(conv_path2_b2, activation=eca_activation)  # Add ECA block here
    conv_path2_b2 = layers.Activation('relu')(conv_path2_b2)

    # Path 3
    maxpool_path_b2 = MaxPooling2D((2, 2), strides=1, padding='same')(block1_output)
    dropout_path_b2 = Dropout(0.25)(maxpool_path_b2)

    # Concatenate all paths at the end of Block 2
    block2_output = Concatenate()([conv_path1_b2, conv_path2_b2, dropout_path_b2])

   
    x = Conv2D(128, (5, 5), padding='same')(block2_output)
    x = eca_block(x, activation=eca_activation)  # Add ECA block here
    x = layers.Activation('relu')(x)
    
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = eca_block(x, activation=eca_activation)  # Add ECA block here
    x = layers.Activation('relu')(x)
    
    x = MaxPooling2D((2, 2))(x)
    # x = MaxPooling2D((2, 2), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)    
    
    x = GlobalAveragePooling2D()(x)


#     x = Flatten()(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(4, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def calculate_map(y_true, y_pred_probs):
    n_classes = y_true.shape[1]
    APs = []
    for i in range(n_classes):
        AP = average_precision_score(y_true[:, i], y_pred_probs[:, i])
        APs.append(AP)
    mAP = np.mean(APs)
    return mAP


def save_metric_details(
    model_name,
    test_acc,
    precision,
    recall,
    f1score,
    mAP_value,
    elapsed_time,
    memory_used,
    result_file_path,
):
    if os.path.exists(result_file_path):
        df_existing = pd.read_csv(result_file_path)
        df_new_row = pd.DataFrame(
            {
                'Model': [model_name],
                'Accuracy': [test_acc],
                'Precision': [precision],
                'Recall': [recall],
                'F1 Score': [f1score],
                'mAP': [mAP_value],
                'Elapsed Time (sec)': [elapsed_time],
                'Memory Used (MB)': [memory_used],
            }
        )

        df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
        df_combined.to_csv(result_file_path, index=False)
    else:
        df_new_row = pd.DataFrame(
            {
                'Model': [model_name],
                'Accuracy': [test_acc],
                'Precision': [precision],
                'Recall': [recall],
                'F1 Score': [f1score],
                'mAP': [mAP_value],
                'Elapsed Time (sec)': [elapsed_time],
                'Memory Used (MB)': [memory_used],
            }
        )

        df_new_row.to_csv(result_file_path, index=False)


train_rgb, train_labels = load_data(dataset_dir, classes, 128, 128)
test_rgb, test_labels = load_data(test_dir, classes, 128, 128)

shuffle_indexes = np.arange(train_rgb.shape[0])
np.random.shuffle(shuffle_indexes)
train_rgb = train_rgb[shuffle_indexes]
train_labels = train_labels[shuffle_indexes]

train_rgb = train_rgb.astype('float32') / 255.0
test_rgb = test_rgb.astype('float32') / 255.0

print(train_rgb.shape)
print(test_rgb.shape)


X_train, X_val, y_train, y_val = train_test_split(train_rgb, train_labels, test_size=0.2, random_state=42)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=4)


start_time = time.time()

with strategy.scope():
    model = build_model_with_eca(eca_activation='sigmoid')
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile( optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint( filepath='/Untitled Folder/Models/New_Model3(softmax).h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    model_early_stopping = tf.keras.callbacks.EarlyStopping( monitor='val_accuracy', min_delta=0, patience=20, restore_best_weights=True, verbose=1)

    history = model.fit( X_train, y_train, epochs=500, validation_data=(X_val, y_val), callbacks=[model_checkpoint, model_early_stopping], verbose=1)

elapsed_time = time.time() - start_time
memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024.0 / 1024.0

test_loss, test_acc = model.evaluate(test_rgb, test_labels_onehot, verbose=0)
print(f"Test accuracy: {test_acc}")


y_pred_probs = model.predict(test_rgb)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(test_labels_onehot, axis=1)

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1score = f1_score(y_true, y_pred, average='macro')
mAP_value = calculate_map(test_labels_onehot, y_pred_probs)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1score}")
print(f"mAP: {mAP_value}")
