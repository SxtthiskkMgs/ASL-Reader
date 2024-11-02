from tensorflow.keras.preprocessing.image import ImageDataGenerator


image_size = (64, 64) 
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0)  
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    'asl_dataset/train', 
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  
)

test_generator = test_datagen.flow_from_directory(
    'asl_dataset/test', 
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  
)

X_train, y_train = train_generator.next() 
X_test, y_test = test_generator.next()  


from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

image_size = (64, 64)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    'asl_dataset/train',  
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'asl_dataset/test',  
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

X_train, y_train = train_generator.next()
X_test, y_test = test_generator.next()

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(26, activation='softmax')
])


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(train_generator,
                    validation_data=test_generator,
                    epochs=6,
                    callbacks=[early_stopping, lr_scheduler])

test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

model.save('my_asl_model.h5')
