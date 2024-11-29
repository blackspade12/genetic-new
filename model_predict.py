import os
import gdown
import pandas as pd
import joblib
import psutil  # To monitor memory usage
import logging

# Set up logging to track memory usage
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define paths and URLs for models (GitHub raw URLs for models)
MODEL_DIR = "models"  # Directory to store models

# Replace with the actual raw URLs of the models on GitHub
MODEL1_URL = "https://github.com/blackspade12/jmedia-genetic-disorder-prediction/blob/main/models/model2.pkl"
MODEL2_URL = "https://github.com/blackspade12/jmedia-genetic-disorder-prediction/blob/main/models/model2.pkl"

MODEL1_PATH = os.path.join(MODEL_DIR, "model1.pkl")
MODEL2_PATH = os.path.join(MODEL_DIR, "model2.pkl")

# Compression option for joblib (can adjust the level for balance between speed and compression)
COMPRESSION = 3  # 0 - no compression, 9 - maximum compression (trade-off between size and speed)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    logging.info(f"Current memory usage: {memory_usage:.2f} MB")

def download_model(model_url, model_path):
    """Download the model file if it doesn't already exist."""
    if not os.path.exists(model_path):
        logging.info(f"Downloading model from {model_url} to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(model_url, model_path, quiet=False)

def compress_model(model, model_path):
    """Compress and save the model using joblib."""
    joblib.dump(model, model_path, compress=COMPRESSION)
    logging.info(f"Model saved with compression at {model_path}")

def init_models():
    """Download models at app startup if they don't exist."""
    download_model(MODEL1_URL, MODEL1_PATH)
    download_model(MODEL2_URL, MODEL2_PATH)
    log_memory_usage()  # Log memory usage after downloading models

def load_models():
    """Load models into memory once at app startup."""
    logging.info("Loading models into memory...")
    model1 = joblib.load(MODEL1_PATH)  # Automatically handles decompression
    model2 = joblib.load(MODEL2_PATH)  # Automatically handles decompression
    log_memory_usage()  # Log memory usage after loading models
    return model1, model2

# Initialize models at app startup
init_models()
model1, model2 = load_models()  # Models are loaded into memory at startup

def predict_genetic_disorder(input_data):
    """Predict genetic disorder using preloaded models."""
    log_memory_usage()  # Log memory usage before prediction
    
    # Convert the input JSON data to a pandas DataFrame
    single_input = pd.DataFrame(input_data)

    # Expected columns from the training dataset
    expected_columns = [
        'White Blood cell count (thousand per microliter)',
        'Blood cell count (mcL)',
        'Patient Age',
        'Father\'s age',
        'Mother\'s age',
        'No. of previous abortion',
        'Blood test result',
        'Gender',
        'Birth asphyxia',
        'Symptom 5',
        'Heart Rate (rates/min',
        'Respiratory Rate (breaths/min)',
        'Folic acid details (peri-conceptional)',
        'History of anomalies in previous pregnancies',
        'Autopsy shows birth defect (if applicable)',
        'Assisted conception IVF/ART',
        'Symptom 4',
        'Follow-up',
        'Birth defects'
    ]

    # Adjust input to match expected columns
    single_input = single_input.reindex(columns=expected_columns, fill_value=0)

    # Make predictions using preloaded models
    final_pred1 = model1.predict(single_input)
    final_pred2 = model2.predict(single_input)

    # Create a DataFrame for predictions
    submission = pd.DataFrame()
    submission['Patient Id'] = [1]  # Replace with dynamic IDs if needed
    submission['Genetic Disorder'] = final_pred1
    submission['Disorder Subclass'] = final_pred2

    # Replace numerical predictions with descriptive strings
    submission['Genetic Disorder'] = submission['Genetic Disorder'].replace(0, 'Mitochondrial genetic inheritance disorders')
    submission['Genetic Disorder'] = submission['Genetic Disorder'].replace(2, 'Single-gene inheritance diseases')
    submission['Genetic Disorder'] = submission['Genetic Disorder'].replace(1, 'Multifactorial genetic inheritance disorders')

    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(0, "Alzheimer's")
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(1, 'Cancer')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(2, 'Cystic fibrosis')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(3, 'Diabetes')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(4, 'Hemochromatosis')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(5, "Leber's hereditary optic neuropathy")
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(6, 'Leigh syndrome')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(7, 'Mitochondrial myopathy')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(8, 'Tay-Sachs')

    # Convert to JSON and return the result
    json_output = submission.to_json(orient='records', lines=True)
    
    log_memory_usage()  # Log memory usage after prediction
    return json_output
