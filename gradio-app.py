import gradio as gr
import requests
from gradio import components

inputs = [
    components.Slider(minimum=-10, maximum=10, label="Attitude Roll"),
    components.Slider(minimum=-10, maximum=10, label="Attitude Pitch"),
    components.Slider(minimum=-10, maximum=10, label="Attitude Yaw"),
    components.Slider(minimum=-10, maximum=10, label="Gravity X"),
    components.Slider(minimum=-10, maximum=10, label="Gravity Y"),
    components.Slider(minimum=-10, maximum=10, label="Gravity Z"),
    components.Slider(minimum=-10, maximum=10, label="Rotation Rate X"),
    components.Slider(minimum=-10, maximum=10, label="Rotation Rate Y"),
    components.Slider(minimum=-10, maximum=10, label="Rotation Rate Z"),
    components.Slider(minimum=-10, maximum=10, label="User Acceleration X"),
    components.Slider(minimum=-10, maximum=10, label="User Acceleration Y"),
    components.Slider(minimum=-10, maximum=10, label="User Acceleration Z")
]

output_text = components.Textbox()

def api_request(attitude_roll, attitude_pitch, attitude_yaw, gravity_x, gravity_y, gravity_z,
                rotationRate_x, rotationRate_y, rotationRate_z, userAcceleration_x,
                userAcceleration_y, userAcceleration_z):
    data = {
        "attitude_roll": attitude_roll,
        "attitude_pitch": attitude_pitch,
        "attitude_yaw": attitude_yaw,
        "gravity_x": gravity_x,
        "gravity_y": gravity_y,
        "gravity_z": gravity_z,
        "rotationRate_x": rotationRate_x,
        "rotationRate_y": rotationRate_y,
        "rotationRate_z": rotationRate_z,
        "userAcceleration_x": userAcceleration_x,
        "userAcceleration_y": userAcceleration_y,
        "userAcceleration_z": userAcceleration_z
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    prediction = response.json()
    formatted_output = "Height: {} cm\nWeight: {} kg\nAge: {}\nGender: {}\nActivity: {}".format(
        prediction["height"], prediction["weight"], prediction["age"], prediction["gender"], prediction["activity"]
    )
    return formatted_output

app = gr.Interface(fn=api_request, inputs=inputs, outputs=output_text, title="Data Prediction")
app.launch()
