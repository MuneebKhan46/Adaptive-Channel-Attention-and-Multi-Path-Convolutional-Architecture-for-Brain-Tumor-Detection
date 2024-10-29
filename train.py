import time
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import build_model_with_eca

# Dataset Paths
train_dir = "~/Dataset/BTD/Training"
test_dir = "~/Dataset/BTD/Testing"
classes = ["glioma", "meningioma", "pituitary", "notumor"]

# Load and Prepare Data
train_data, train_labels = load_data(train_dir, classes)
train_data = train_data.astype('float32') / 255.0  # Normalize the data

X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))

# Build and Compile Model
model = build_model_with_eca()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("model.h5", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

# Train the Model
start_time = time.time()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500,
                    callbacks=[checkpoint, early_stopping], verbose=1)
print(f"Training time: {time.time() - start_time:.2f} seconds")
