import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import load_data
from model import build_model_with_eca

dataset_path = "~/brain-tumor-mri-dataset/Training"
classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

data, labels = load_data(dataset_path)
data = data.astype('float32') / 255.0  # Normalize pixel values

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))

model = build_model_with_eca()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("model.h5", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[checkpoint, early_stopping], verbose=1)
