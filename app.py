from fastapi import FastAPI
from joblib import load
from src.model import InferenceModel 
from datetime import datetime
from mangum import Mangum
from loguru import logger
from src.dynamo_db import UncertaintyDynamoDB


logger.info("Start Prepare Web Service")
app = FastAPI()
handler = Mangum(app=app)

logger.info("Prepare Inference Model")
model = InferenceModel()
uncertainty_db = UncertaintyDynamoDB(table_name='uncertainty')

@app.get("/")
async def predict(text: str):
    logger.info(f"Running Prediction Process. Input: {text}")
    result_predict = model.predict(text=text)
    # Get the current timestamp and format it
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Add timestamp
    result_predict["timestamp"] = timestamp 
    logger.info(f"Prediction Success. Result: {result_predict}")

    uncertainty_db.write_event(
        data=result_predict
    )

    return result_predict

if __name__ == "__main__":
    import uvicorn
    logger.info("Running Main App")
    uvicorn.run(app, host="0.0.0.0", port=8000)