import os
import gdown
import pandas as pd
import joblib
import tempfile  # Added for handling temp directory

# Function to download model from Google Drive using gdown
def download_model(model_url, model_path):
    if not os.path.exists(model_path):  # Check if the model already exists
        gdown.download(model_url, model_path, quiet=False)
    return joblib.load(model_path)

# Function to load model1 on demand
def load_model1():
    model1_url = "https://drive.google.com/uc?id=1cATLRCX35rOPEo5DlrhB8QKZjjcrf5JC"  # Model 7 file ID
    model1_path = os.path.join(tempfile.gettempdir(), "model1.pkl")  # Use a temp directory
    return download_model(model1_url, model1_path)

# Function to load model2 on demand
def load_model2():
    model2_url = "https://drive.google.com/uc?id=186kGZFhB1rSPLqqVyKEgO-JcSH0mqlCw"  # Model 15 file ID
    model2_path = os.path.join(tempfile.gettempdir(), "model2.pkl")  # Use a temp directory
    return download_model(model2_url, model2_path)

def predict_genetic_disorder(input_data):
    # Load models on demand
    model1 = load_model1()
    model2 = load_model2()

    # Convert the input JSON data to a pandas DataFrame
    single_input = pd.DataFrame(input_data)

    # Columns from the training dataset
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

    # Adjust the input to match expected columns
    single_input = single_input.reindex(columns=expected_columns, fill_value=0)

    # Make predictions
    final_pred1 = model1.predict(single_input)
    final_pred2 = model2.predict(single_input)

    # Create a DataFrame for the predictions
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
    return json_output
