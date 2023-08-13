from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json

app = FastAPI()


class SensorData(BaseModel):
    attitude_roll: float
    attitude_pitch: float
    attitude_yaw: float
    gravity_x: float
    gravity_y: float
    gravity_z: float
    rotationRate_x: float
    rotationRate_y: float
    rotationRate_z: float
    userAcceleration_x: float
    userAcceleration_y: float
    userAcceleration_z: float


numerical_pipeline = joblib.load(
    "final_pipeline_numerical")
categorical_pipeline = joblib.load(
    "final_pipeline_categorical")


@app.post("/test")
def test():
    return "TEST"

@app.post("/predict")
def prediction(input_parameters: SensorData):
    input_data = input_parameters.json()
    input_df = json.loads(input_data)

    attitude_roll = input_df["attitude_roll"]
    attitude_pitch = input_df["attitude_pitch"]
    attitude_yaw = input_df["attitude_yaw"]
    gravity_x = input_df["gravity_x"]
    gravity_y = input_df["gravity_y"]
    gravity_z = input_df["gravity_z"]
    rotationRate_x = input_df["rotationRate_x"]
    rotationRate_y = input_df["rotationRate_y"]
    rotationRate_z = input_df["rotationRate_z"]
    userAcceleration_x = input_df["userAcceleration_x"]
    userAcceleration_y = input_df["userAcceleration_y"]
    userAcceleration_z = input_df["userAcceleration_z"]

    features_list = [attitude_roll, attitude_pitch, attitude_yaw, gravity_x, gravity_y, gravity_z,
                       rotationRate_x, rotationRate_y, rotationRate_z, userAcceleration_x,
                       userAcceleration_y, userAcceleration_z]

    categorical_predictions = categorical_pipeline.predict([features_list])[0]
    numerical_predictions = numerical_pipeline.predict([features_list])[0]

    activities = ["Downstairs", "Jogging",
                  "Sitting", "Standing", "Upstairs", "Walking"]
    gender = ["Female", "Male"]

    return {
        "height": "{:.2f}".format(numerical_predictions[1]),
        "weight": "{:.2f}".format(numerical_predictions[0]),
        "age": str(round(numerical_predictions[2])),
        "gender": gender[categorical_predictions[0]],
        "activity": activities[categorical_predictions[1]]
    }
