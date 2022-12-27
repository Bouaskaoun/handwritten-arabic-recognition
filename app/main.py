from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI(
    title='Handwritten Character Recognition',
    version='1.0',
    description='A simple API to predict handwritten characters'
    )

# Load the trained random forest classifier from a pickle file
model = joblib.load("HandwrittenModel.joblib")


class Image(BaseModel):
    # Declare the image field as a list of floats
    image: list

# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}

# ML API endpoint for making prediction aganist the request received from client
@app.post("/predict")
def predict(image: Image):
    label_map = {
        'ain':0,
        'alif':1,
        'beh':2,
        'dal':3,
        'feh':4,
        'heh':5,
        'jeem':6,
        'kaf':7,
        'lam':8,
        'meem':9,
        'noon':10,
        'qaf':11,
        'raa':12,
        'sad':13,
        'seen':14,
        'tah':15,
        'waw':16,
        'yaa':17
        }
    # Make prediction using the model
    value = model.predict([image.image])[0]
    key = [k for k, v in label_map.items() if v == value]
    prediction = 'The class predicted for image is ' + key[0]
    return {"prediction": prediction}

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)