import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import build_model_with_eca

num_classes = 4
data, labels = load_data()
data = data.astype('float32') / 255.0  # Normalize pixel values

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)


model = build_model_with_eca()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


callbacks = [tf.keras.callbacks.ModelCheckpoint("~/model.h5", save_best_only=True, monitor='val_accuracy'),
             tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True) ]

start_time = time.time()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=callbacks)
print(f"Training completed in {time.time() - start_time:.2f} seconds.")
