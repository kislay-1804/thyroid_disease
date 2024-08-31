import pandas as pd
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing, clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import pickle

class ModelTraining:
    def __init__(self):
        self.log_writer = logger.AppLogger()
        self.file_object = open(r"path\to\ModelTrainingLog.txt", 'a+')

    def train_model(self, csv_file_path):
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Load data from CSV
            data = pd.read_csv(csv_file_path)
            
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            data = preprocessor.replace_invalid_values_with_null(data)
            data = preprocessor.encode_categorical_values(data)
            X, Y = preprocessor.separate_label_feature(data, label_column_name='Class')

            if preprocessor.is_null_present(X):
                X = preprocessor.impute_missing_values(X) 

            X, Y = preprocessor.handle_imbalance_dataset(X, Y)
            
            kmeans = clustering.KMeansClustering(self.file_object, self.log_writer) 
            number_of_clusters = kmeans.elbow_plot(X)
            X = kmeans.create_clusters(X, number_of_clusters)
            X['Labels'] = Y

            # Store the feature names
            feature_names = X.drop(['Labels', 'Cluster'], axis=1).columns.tolist()
            
            # Initialize model finder
            model_finder = tuner.ModelFinder(self.file_object, self.log_writer)

            for cluster in X['Cluster'].unique():
                cluster_data = X[X['Cluster'] == cluster]
                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_data.drop(['Labels', 'Cluster'], axis=1), 
                    cluster_data['Labels'], 
                    test_size=0.25, random_state=42)

                # Find the best model
                best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)

                # Save the model as a pickle file
                file_op = file_methods.FileOperation(self.file_object, self.log_writer)
                model_filename = f"{best_model_name}_cluster_{cluster}.pkl"
                with open(model_filename, 'wb') as model_file:
                    pickle.dump({'model': best_model, 'feature_names': feature_names}, model_file)

            self.log_writer.log(self.file_object, 'Successful End of Training')
        except Exception as e:
            self.log_writer.log(self.file_object, f'Unsuccessful End of Training: {str(e)}')
            raise
        finally:
            self.file_object.close()