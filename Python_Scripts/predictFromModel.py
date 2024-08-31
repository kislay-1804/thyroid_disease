import pandas as pd
import pickle
from file_operations import file_methods
from data_preprocessing import preprocessing
from application_logging import logger

class Prediction:
    def __init__(self, log_path, model_path):
        self.log_path = log_path
        self.model_path = model_path
        self.log_writer = logger.App_Logger()

    def predict_from_model(self, form_data):
        """Predict the outcome based on the provided form data."""
        try:
            self.log_writer.log(self.log_path, 'Start of Prediction')

            # Convert form data to DataFrame
            data = pd.DataFrame([form_data])
            self.log_writer.log(self.log_path, f'Form data converted to DataFrame: {data}')

            # Initialize preprocessor and preprocess data
            preprocessor = preprocessing.Preprocessor(self.log_path, self.log_writer)
            data = preprocessor.replaceInvalidValuesWithNull(data)
            data = preprocessor.encodeCategoricalValuesPrediction(data)

            if preprocessor.is_null_present(data):
                data = preprocessor.impute_missing_values(data)

            # Load KMeans model and predict clusters
            file_loader = file_methods.FileOperation(self.log_path, self.log_writer)
            kmeans = file_loader.load_model('KMeans')
            data['clusters'] = kmeans.predict(data)
            self.log_writer.log(self.log_path, f'Clusters predicted: {data["clusters"].tolist()}')

            # Load the encoder and predict using cluster-specific models
            with open(self.model_path, 'rb') as file:
                encoder = pickle.load(file)

            result = []
            for cluster in data['clusters'].unique():
                cluster_data = data[data['clusters'] == cluster].drop(['clusters'], axis=1)
                model = file_loader.load_model(file_loader.find_correct_model_file(cluster))

                # Predict and decode the results
                predictions = model.predict(cluster_data)
                decoded_results = encoder.inverse_transform(predictions)
                result.extend(decoded_results)

            # Return the prediction result
            prediction_result = result[0] if result else 'No prediction result'

            self.log_writer.log(self.log_path, 'End of Prediction')
            return prediction_result

        except Exception as ex:
            self.log_writer.log(self.log_path, f'Error during prediction: {ex}')
            raise

if __name__ == "__main__":
    # Define file paths
    log_path = r"path\to\Prediction_Log.txt"
    model_path = r"path\to\Thyroid_model.pickle"

    # Hardcoded test data for validation
    form_data = {
        'age': '45',
        'sex': 'Female',
        'on_thyroxine': 'True',
        'query_on_thyroxine': 'False',
        'on_antithyroid_medication': 'True',
        'sick': 'False',
        'pregnant': 'False',
        'thyroid_surgery': 'False',
        'I131_treatment': 'False',
        'query_hypothyroid': 'True',
        'query_hyperthyroid': 'False',
        'lithium': 'False',
        'goitre': 'True',
        'tumor': 'False',
        'hypopituitary': 'False',
        'psych': 'False',
        'TSH_measured': 'True',
        'TSH': '2.0',
        'T3_measured': 'True',
        'T3': '1.0',
        'TT4_measured': 'True',
        'TT4': '100.0',
        'T4U_measured': 'True',
        'T4U': '0.5',
        'FTI_measured': 'True',
        'FTI': '150.0',
        'referral_source': 'SVHC'
    }

    # Perform prediction
    predictor = Prediction(log_path, model_path)
    try:
        result = predictor.predict_from_model(form_data)
        print(f'Prediction Result: {result}')
    except Exception as e:
        print(f'An error occurred: {e}')