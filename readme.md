# Ocular Disease Recognition Project

## Overview
This project builds a machine learning model using Python and TensorFlow to analyze eye images for ocular disease recognition. The model classifies images into various categories (such as Normal, Diabetes, Glaucoma, etc.) based on features extracted from eye images.

*** IMPORTANT: This Project was converted from a Notebook for local use and has not been thoroughly tested.

## Tools and Technologies
- **Language:** Python 3.13.1
- **Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-Learn
- **IDE:** PyCharm
- **Environment Management:** pip/virtualenv

## Setup and Installation

1. **Clone the Repository:**  
   ```bash
   git clone <repository-link>
   cd Odir_Classification
   ```

2. **Create and Activate a Virtual Environment:**  
   ```bash
   python -m venv env
   source env/bin/activate  # For Unix systems
   # or
   env\Scripts\activate     # For Windows
   ```

3. **Install Required Libraries:**  
   ```bash
   pip install tensorflow pandas scikit-learn pillow python-dotenv
   ```

4. **Dataset Setup:**  
   Download and extract the dataset from [Kaggle Ocular Disease Recognition](https://www.kaggle.com/...).  
   Place the extracted folder in the project directory.

   > **Note:**  
   > The image files from the Kaggle dataset should be stored in the directory specified by the `IMAGES_BASE_DIR` environment variable.  
   > By default, this is set to:  
   > `data/Training/ODIR-5K/ODIR-5K/Training Images`  
   > If your data is stored elsewhere, update the `IMAGES_BASE_DIR` variable in your `.env` file accordingly.
   
5. **Environment Variables Setup:**  
   Create a `.env` file in the root directory with the following variables to override default paths:
   ```dotenv
   CSV_PATH=data/full_df.csv
   PY_SCRIPT=model_prediction_V2.py
   HISTORY_PATH=data/training_history.csv
   MODEL_PATH=models/model_v2.tf
   LOCAL_IMAGE=path/to/your/image.jpg
   ```
   If the `IMAGES_BASE_DIR` variable is not provided in the `.env` file, the script will fallback to the default image directory.
- **LOCAL_IMAGE:**  
  Optional. Define the path to a predefined image that the prediction notebook will use automatically. If this variable is set and the image exists, the notebook will process this image directly for prediction instead of using the interactive uploader.

If `LOCAL_IMAGE` is not provided or the specified file cannot be found, the notebook will revert to the default behavior and display an interactive image uploader widget.

## Usage

### Training the Model
- **Main Script:**  
  The `main.py` script is dedicated to training the model. It loads the dataset metadata (using the CSV path defined in your `.env` file or a default), preprocesses the image paths and labels, splits the data into training and validation sets, creates image generators, and trains the model using the ResNet50 architecture with additional dense layers.

  To train the model, simply run:
  ```bash
  python main.py
  ```
  After training, the model is saved to the location defined by the `MODEL_PATH` environment variable in your `.env` file (or a default path if not set).

### Utilizing the Trained Model
- **Notebook:**  
  A separate notebook is provided to demonstrate how to utilize the trained model. This notebook walks through loading the saved model and applying it for inference on new data. The notebook explains each step of the process, from loading the model to interpreting the prediction results.

## Model Details
The model is based on the ResNet50 architecture with pre-trained weights from ImageNet. Custom dense layers are added to the network to fine-tune the model for ocular disease recognition. Key features include:
- Data augmentation using image generators for robust training.
- Fine-tuning of the last layers of the ResNet50 architecture.
- Environment variable support for flexible configuration of paths.

## Additional Notes
- Ensure the environment variables in your `.env` file are correctly set according to your local file structure.
- The model training process is separated from model utilization to keep training and inference workflows distinct.
- For further customization, you can modify the image augmentation settings and fine-tuning configurations in `main.py`.
