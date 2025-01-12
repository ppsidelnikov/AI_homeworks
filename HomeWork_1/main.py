
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List
import pandas as pd
import re

import re
def extract_number_from(value):
    if pd.notnull(value):
        result = re.search(r'(\d+)', value)
        if result is not None:
            return int(result.group(0))
    return np.nan

def parse_torque(value):
  if pd.isnull(value):
        return np.nan, np.nan
  torque_match = re.search(r'(\d+\.?\d*)\s*(?:Nm|kgm)', value, re.IGNORECASE)
  if torque_match:
      torque = float(torque_match.group(1))
  else:
      torque = np.nan

  rpm_match = re.search(r'(\d+)(?!.*\d)', value, re.IGNORECASE)
  if rpm_match:
    max_torque_rpm = int(rpm_match.group(1).replace(',', ''))

  else:
      max_torque_rpm = np.nan

  return torque, max_torque_rpm


app = FastAPI()
model = joblib.load('model_weights.pkl')

# Пример класса объекта автомобиля
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]


def process_item(self) -> List[float]:
  torque, max_torque_rpm = parse_torque(self.torque)
  mileage = extract_number_from(self.mileage)
  engine = extract_number_from(self.engine)
  max_power = extract_number_from(self.max_power)

  vector = [[
      self.year,
      self.km_driven,
      mileage,
      engine,
      max_power,
      torque,
      max_torque_rpm,
      self.seats
  ]]

  return vector



def predict(item: Item) -> float:
    try:
        prediction = model.predict(process_item(item))
        return prediction
    except:
        pass
    


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    prediction = predict(item)
    return prediction

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    predictions = [predict(item) for item in items.objects]
    return predictions

@app.post("/predict_from_csv")
async def predict_from_csv(file: UploadFile = File(...)) -> FileResponse:
    df = pd.read_csv(file.file)
    mis = df.columns[df.isnull().any()].tolist()
    for col in mis:
        med = df[col].median()
        df[col] = df[col].fillna(med)
    predictions = []
    for index, row in df.iterrows():
        item = Item(**row)
        prediction = predict(item)
        predictions.append(prediction)

    df['Predicted_Price'] = predictions

    # Записать выходной файл CSV
    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)

    return FileResponse(output_file, filename="predictions_with_prices.csv")