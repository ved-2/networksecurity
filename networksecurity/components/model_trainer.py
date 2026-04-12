import os
import sys
import pickle
import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='ved-2', repo_name='networksecurity', mlflow=True)


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/ved-2/networksecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="ved-2"
os.environ["MLFLOW_TRACKING_PASSWORD"]="7d1974b2d8f3107d2e9760e1814128b77dbb0b71"

import skops.io as sio
from mlflow.exceptions import MlflowException

class ModelTester:
    def __init__(self, model_path):
        """
        Initialize the ModelTester with the path to the trained model.
        :param model_path: Path to the .pkl file containing the trained model.
        """
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        Load the trained model from the .pkl file.
        :return: Loaded model object.
        """
        try:
            with open(self.model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def preprocess_data(self, data):
        """
        Preprocess the new data to match the training data format.
        :param data: Raw input data as a Pandas DataFrame.
        :return: Preprocessed data ready for prediction.
        """
        # Add preprocessing steps here (e.g., scaling, encoding)
        return data

    def predict(self, data):
        """
        Predict the class of the new data using the trained model.
        :param data: Preprocessed data as a Pandas DataFrame.
        :return: Predictions as a list.
        """
        try:
            predictions = self.model.predict(data)
            return predictions
        except Exception as e:
            raise Exception(f"Error during prediction: {e}")

    def classify(self, predictions):
        """
        Map predictions to labels (e.g., Safe, Harmful, Suspicious).
        :param predictions: List of predictions.
        :return: List of classification labels.
        """
        label_mapping = {0: "Safe", 1: "Harmful", 2: "Suspicious"}
        return [label_mapping.get(pred, "Unknown") for pred in predictions]

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self, best_model, classificationmetric):
        mlflow.set_registry_uri("https://dagshub.com/ved-2/networksecurity.mlflow/")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            # Save the model using skops
            sio.dump(best_model, "model.skops")

            # Log the skops model file to MLflow
            mlflow.log_artifact("model.skops", artifact_path="model")

            try:
                # Check if the model is already registered
                client = mlflow.tracking.MlflowClient()
                registered_model = client.get_registered_model("BestModel")
                logging.info(f"Model 'BestModel' already registered. Creating a new version.")
            except MlflowException:
                # Register the model if it doesn't exist
                mlflow.sklearn.log_model(best_model, name="model", registered_model_name="BestModel")
        
        
    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        
        ## Track the experiements with mlflow
        self.track_mlflow(best_model,classification_train_metric)


        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        self.track_mlflow(best_model,classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
        
        #model pusher

        save_object("final_model/model.pkl",best_model)
        
        # Save the model using skops
        sio.dump(best_model, "model.skops")

        # Log the skops model file to MLflow
        mlflow.log_artifact("model.skops", artifact_path="model")

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)

from mlflow.exceptions import MlflowException

try:
    # Check if the model is already registered
    client = mlflow.tracking.MlflowClient()
    registered_model = client.get_registered_model("BestModel")
    logging.info(f"Model 'BestModel' already registered. Creating a new version.")
except MlflowException:
    # Register the model if it doesn't exist
    mlflow.sklearn.log_model(best_model, name="model", registered_model_name="BestModel")