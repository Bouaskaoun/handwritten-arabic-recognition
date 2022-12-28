# Arabic Handwritten Characters Classification

This project aims to classify Arabic handwritten characters using the Random Forest classifier and FastAPI for the API and Streamlit for the frontend.

## Requirements

To run this project, you will need to have the following dependencies installed:

- Python 3.7 or higher
- FastAPI
- Streamlit
- Scikit-learn
- Numpy
- cv2
- Pillow

You can install these dependencies using `pip`. For example:

```bash
pip install fastapi streamlit scikit-learn numpy opencv-python pillow
```

## Usage

To start the API and the frontend, run the following commands:

```bash
# Start the API
uvicorn main:app --reload

# Start the frontend
streamlit run app.py
```

The frontend will be available at **http://localhost:8501** and the API will be available at **http://localhost:8000**.

## Data

- The data used in this project is a subset of the [Arabic Handwritten Alphabets Dataset](https://data.mendeley.com/datasets/2h76672znt/1).
- The dataset contains 65 different Arabic alphabets (with variations on begin, middle, end and regular alphabets).
- The dataset was collected anonymously from 82 different users. Each user was asked to write each alphabet 10 times.
- A userid uniquely but anonymously identifies the writer of each alphabet. In total, the dataset consists of 53199 alphabet images.

## Model

The model used in this project is a Random Forest classifier, trained on the dataset mentioned above. The model was trained using scikit-learn's implementation of the
Random Forest classifier, with default hyperparameters.

## API

The API allows you to classify images of Arabic handwritten characters by sending a POST request to the `/predict` endpoint. The request should include the image as a
list of floats in the image field of the request body.

The API will return a JSON object with the following fields:

- `prediction`: The predicted class of the image (string)

## Frontend

The frontend is a simple web app built using Streamlit. It allows you to select an image and send it to the API for classification. The frontend displays the
prediction returned by the API.

## Credits

The data used in this project was obtained from : https://data.mendeley.com/datasets/2h76672znt/1
