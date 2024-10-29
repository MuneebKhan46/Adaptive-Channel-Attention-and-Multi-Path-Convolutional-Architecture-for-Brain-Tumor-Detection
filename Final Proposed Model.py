#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import time
import cv2
import numpy as np
import pandas as pd
import psutil
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tracemalloc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Concatenate, Softmax


# In[10]:


strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


# In[ ]:


result_file_path = "/Untitled Folder/Results.csv"

dataset_dir = "/Untitled Folder/Dataset Folder/BTD/Training"
test_dir = "/Untitled Folder/Dataset Folder/BTD/Testing"

classes = ["glioma", "meningioma", "pituitary", "notumor"]


# In[ ]:


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


# In[ ]:


def eca_block(input_feature, k_size=3,  activation='swish'):
    channel = input_feature.get_shape().as_list()[-1]
    if channel is None:
        raise ValueError('The channel dimension of the input_feature must be defined.')

    k_size = max(int(abs((np.log2(channel) + 1) / 2)), 1)
    k_size = k_size if k_size % 2 else k_size + 1

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((channel, 1))(avg_pool)
    conv1d = layers.Conv1D(1, kernel_size=k_size, padding='same', use_bias=False)(avg_pool)
    conv1d = layers.Activation(activation)(conv1d)
    conv1d = layers.Reshape((1, 1, channel))(conv1d)

    return layers.Multiply()([input_feature, conv1d])



# In[ ]:


def create_block1(filters, kernel_size, strides=1):
    block = models.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2),  strides=1, padding='same'),
        layers.BatchNormalization()
    ])
    return block

def create_block2(filters, kernel_size, strides=1):
    block = models.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu'),
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2),  strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25)
    ])
    return block

def create_main_block(input_tensor, filters, kernel_size1, kernel_size2):
    path1 = create_block1(filters, kernel_size1)(input_tensor)
    path2 = create_block1(filters, kernel_size2)(input_tensor)
    path3 = create_block2(filters, kernel_size2)(input_tensor)
    return layers.Concatenate()([path1, path2, path3])



# In[ ]:


def attention_block(x, ratio=8):
    channel = x.get_shape().as_list()[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    shared_layer_one = layers.Dense(channel//ratio, activation='swish', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool = shared_layer_two(shared_layer_one(avg_pool))
    max_pool = shared_layer_two(shared_layer_one(max_pool))
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.Multiply()([x, cbam_feature])


# In[11]:


def build_model_with_eca(input_shape=(224, 224, 3), num_classes=4, eca_k_size=3, eca_activation='softmax'):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    x = create_main_block(x, 32, 1, 3)
    
    
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    x = eca_block(x, k_size=eca_k_size, activation=eca_activation)
    
    x = create_main_block(x, 64, 1, 3)
    
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)  
   
    x = eca_block(x, k_size=eca_k_size, activation=eca_activation)
  
    x = create_main_block(x, 128, 3, 5)    

    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    
    x = eca_block(x, k_size=eca_k_size, activation=eca_activation)
    
    
    x = attention_block(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model



model = build_model_with_eca()

model.summary()


# In[9]:


def calculate_map(y_true, y_pred_probs):
    n_classes = y_true.shape[1]
    APs = []
    for i in range(n_classes):
        AP = average_precision_score(y_true[:, i], y_pred_probs[:, i])
        APs.append(AP)
    mAP = np.mean(APs)
    return mAP


# In[10]:


def save_metric_details(model_name, test_acc, precision, recall, f1_score, mAP, elapsed_time, memory_used, infer_time, result_file_path):
    if os.path.exists(result_file_path):
        df_existing = pd.read_csv(result_file_path)
        df_new_row = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [test_acc],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1_score],
            'mAP': [mAP],
            'Elapsed Time (sec)': [elapsed_time],
            'Memory Used (MB)': [memory_used],
            'Inference Time': [infer_time]
        })

        df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
        df_combined.to_csv(result_file_path, index=False)
    else:
        df_new_row = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [test_acc],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1_score],
            'mAP': [mAP],
            'Elapsed Time (sec)': [elapsed_time],
            'Memory Used (MB)': [memory_used],
            'Inference Time': [infer_time]
        })

        df_new_row.to_csv(result_file_path, index=False)


# In[11]:


train_rgb, train_labels = load_data(dataset_dir, classes, 224, 224)
test_rgb, test_labels = load_data(test_dir, classes, 224, 224)


shuffle_indexes = np.arange(train_rgb.shape[0])
np.random.shuffle(shuffle_indexes)
train_rgb = train_rgb[shuffle_indexes]
train_labels = train_labels[shuffle_indexes]



train_rgb = train_rgb.astype('float32') / 255.0
test_rgb = test_rgb.astype('float32') / 255.0



print(train_rgb.shape)
print(test_rgb.shape)


# In[12]:


X_train, X_val, y_train, y_val = train_test_split(train_rgb, train_labels, test_size=0.2, random_state=42)


y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=4)


# In[ ]:





# In[18]:


start_time = time.time()
with strategy.scope():
    model = build_model_with_eca()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/Untitled Folder/Models/Proposed Model2.h5', save_best_only=True,
                                                          monitor='val_accuracy', mode='max',verbose=1)

    model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0,patience=20,restore_best_weights=True)


    history= model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), callbacks=[model_checkpoint, model_early_stopping], verbose=1)


elapsed_time = time.time() - start_time
memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024.0 / 1024.0


# In[14]:


test_loss, test_accuracy = model.evaluate(test_rgb, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[15]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

y_true = []
y_pred = []
y_pred_probs = []

start_test = time.time()

preds = model.predict(test_rgb, batch_size=1)

end_test = time.time() - start_test

y_pred_probs.extend(preds)
y_pred.extend(np.argmax(preds, axis=1))


y_true.extend(np.argmax(test_labels, axis=1))


y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_probs = np.array(y_pred_probs)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')


mAP = calculate_map(tf.keras.utils.to_categorical(y_true, 4), y_pred_probs)
inference_time = end_test / len(test_rgb)


model_name = "Proposed Methodology"

print(model_name, accuracy, precision, recall, f1, mAP, elapsed_time, memory_used, inference_time)

save_metric_details(model_name, accuracy, precision, recall, f1, mAP, elapsed_time, memory_used, inference_time, result_file_path)


# In[16]:


conf_matrix = confusion_matrix(y_true, y_pred)

updated_classes = ['Glioma', 'Meningioma', 'Pituitary', 'No-Tumor']

plt.figure(figsize=(8, 6))

ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=updated_classes, yticklabels=updated_classes, annot_kws={"size": 14})

ax.set_xlabel('Predicted Label', fontsize=12, weight='bold')
ax.set_ylabel('True Label', fontsize=12, weight='bold')

plt.title('Confusion Matrix', fontsize=13, weight='bold')

plt.show()


# In[17]:


y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(classes))

plt.figure(figsize=(10, 8))
for i, class_name in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {class_name} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Multi-Class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


a =pd.read_csv("/Untitled Folder/Results.csv")
a.head()


# In[ ]:




