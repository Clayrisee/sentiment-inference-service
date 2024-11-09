
import boto3
import uuid
from .utils import convert_floats_to_decimals
from loguru import logger
import pandas as pd

class UncertaintyDynamoDB:
    def __init__(self, table_name: str) -> None:
        logger.info(f"Setting UncertaintyDynamoDB with name: {table_name}")
        self.dynamo_db = boto3.resource('dynamodb')
        self.uncertainty_table = self.dynamo_db.Table(table_name)
        self.table_name = table_name
    
    def write_event(self, data: dict):
        try:
            logger.info(f"Write event: {data} to Table: {self.table_name}")
            data = convert_floats_to_decimals(data)
            response = self.uncertainty_table.put_item(
            Item={ 
                'id': str(uuid.uuid4()),
                'timestamp': data['timestamp'], 
                'text': data['text'], 
                'prediction': data['prediction'], 
                'confidence': str(data['confidence']),
                'uncertainty_score': data['uncertainty_score'],
                'is_uncertain': data['is_uncertain'] 
            }
            )
            logger.info(f"Write Event Success. Response:{response}")
        except Exception as e:
            logger.error(f"Error write event. Error: {e}")
    
    def fetch_data(self) -> pd.DataFrame:
        try:
            response = self.uncertainty_table.scan()
            data = response["Items"]
            while 'LastEvaluatedKey' in response:
                response = self.uncertainty_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                data.extend(response['Items'])
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return pd.DataFrame()