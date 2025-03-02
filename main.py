# main.py
import argparse
import os
import pandas as pd
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

# Constants
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS = 30
CONDITIONS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']  # The condition columns in the CSV


DEFAULT_IMAGES_BASE_DIR = os.getenv('IMAGES_BASE_DIR', 'input/ODIR-5K/ODIR-5K/Training Images')
DEFAULT_CSV_FILEPATH = os.getenv('CSV_PATH', 'input/full_df.csv')


# Function to load CSV data
def load_csv_data(filepath):
    return pd.read_csv(filepath)


# Function to preprocess CSV data
def preprocess_csv_data(df, base_image_dir):
    # Extract the condition columns for one-hot encoded labels
    labels_df = df[CONDITIONS]

    # Adjust file paths in the DataFrame
    df['Left-Fundus'] = df['Left-Fundus'].apply(lambda x: os.path.join(base_image_dir, x))
    df['Right-Fundus'] = df['Right-Fundus'].apply(lambda x: os.path.join(base_image_dir, x))

    # Combine left and right fundus images into a single list for simplicity.
    # If you need separate handling for left and right images, add extra logic.
    all_file_paths = df['Left-Fundus'].tolist() + df['Right-Fundus'].tolist()
    all_labels = labels_df.values.tolist() + labels_df.values.tolist()

    return all_file_paths, all_labels


def build_model(num_conditions):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)  # New dense layer
    x = Dense(512, activation='relu')(x)  # Another dense layer
    # x = Dropout(0.5)(x)  # Uncomment if dropout is desired

    predictions = Dense(num_conditions, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Fine-tuning: freeze early layers and train later layers
    for layer in base_model.layers[:143]:
        layer.trainable = False
    for layer in base_model.layers[143:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Main function that performs training using the specified CSV and image directory
def main(csv_filepath, images_base_dir):
    # Load and preprocess CSV data
    csv_data = load_csv_data(csv_filepath)
    filepaths, labels = preprocess_csv_data(csv_data, images_base_dir)

    # Convert lists to a pandas Series and DataFrame
    filepaths_series = pd.Series(filepaths, name='filepath')
    labels_df = pd.DataFrame(labels, columns=CONDITIONS)

    # Combine into a single DataFrame for the image data generator
    combined_df = pd.concat([filepaths_series, labels_df], axis=1)

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        combined_df['filepath'], combined_df[CONDITIONS], test_size=0.2, random_state=42
    )

    # Prepare DataFrames for generators
    train_df = pd.concat([x_train, y_train], axis=1)
    val_df = pd.concat([x_val, y_val], axis=1)

    # Create image data generators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=None,  # Absolute paths are provided in 'filepath'
        x_col='filepath',
        y_col=CONDITIONS,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='raw'
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=None,
        x_col='filepath',
        y_col=CONDITIONS,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='raw'
    )

    # Build and compile the model
    model = build_model(len(CONDITIONS))

    # Train the model
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

    # Save the model
    model_save_path = os.getenv('MODEL_PATH', 'trained-models/model.tf')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ocular Disease Recognition Training")
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV_FILEPATH,
        help="Path to the CSV file with metadata (default: %(default)s)"
    )
    parser.add_argument(
        "--image_dir", type=str, default=DEFAULT_IMAGES_BASE_DIR,
        help="Base directory for training images (default: %(default)s)"
    )
    args = parser.parse_args()
    main(args.csv, args.image_dir)
