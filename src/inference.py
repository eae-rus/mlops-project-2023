import pandas as pd
import mlflow
import os
from dotenv import load_dotenv
import unicorn
from fastapi import FastAPI, File, UploadFile


load_dotenv()

app = FastAPI()

class Model:
    def __init__(self, name, stage):

        self.model = mlflow.pyfunc.load_model(f'models:/{name}/{stage}')

    def predict(self, data):

        predictions = self.model.predict(data)

        return predictions
    

model = Model('mlp', 'Staging')


@app.posr('/invocations')
async def create_upload_file(file: UploadFile = File(...)):
    if file.filename.endwith('.csv'):
        with open(file.filename, 'wb') as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)

        return list(model.predict(data))
    
    else:
        raise Exception(detail="Invalid file format")
    
if os.getenv('AWS_ACCESS_KEY_ID') is None or os.getenv('AWS_SECRET_ACCESS_KEY') is None:
    exit(1)