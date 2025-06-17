import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    r"TRAIN_DATASET",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    r"VALIDATION_DATASET",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Build model function
def build_densenet_model():
    # Load DenseNet-121 as the base model
    base_model = DenseNet121(weights='imagenet', include_top=False)
    base_model.trainable = False  # Freeze base model layers
    
    # Define input tensor
    input_tensor = Input(shape=(224, 224, 3))
    
    # Add base model
    x = base_model(input_tensor, training=False)
    
    # Add custom layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(train_generator.num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_tensor, outputs=output)
    return model

# Create the DenseNet-121 model
densenet_model = build_densenet_model()

# Compile the model
densenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = densenet_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=5,  # Adjust based on resources
)

# Save the trained model
save_path = r"SAVE_PATH"
densenet_model.save(save_path)

print(f"Model saved to {save_path}")
