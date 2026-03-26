from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
import sys
from networksecurity.components.data_validation import DataValidationConfig,DataValidation

if __name__ == "__main__":
    try:
      trainingpipelineconfig=TrainingPipelineConfig()
      
      dataingestionconfig = DataIngestionConfig(training_pipeline_config=trainingpipelineconfig)
      dataingestion=DataIngestion(dataingestionconfig)
      
      
      logging.info("Initiate the data ingestion")
      dataingestionartifact=dataingestion.initiate_data_ingestion()
      print(dataingestionartifact)
      
      
      data_validation_config=DataValidationConfig(trainingpipelineconfig)
      data_validation=DataValidation(dataingestionartifact,data_validation_config)
      
      logging.info("Initiate the dat Validation")
      
      data_validation_artifact=data_validation.initiate_data_validation()
      logging.info("Initiate the data validation")
      
      print(data_validation_artifact)


    except Exception as e:
        raise NetworkSecurityException(e, sys)