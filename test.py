import requests
import json

url = "http://127.0.0.1:8000/predict"

data = {
    'attitude_roll': 1.528132,
    'attitude_pitch': -0.733896,
    'attitude_yaw': 0.696372,
    'gravity_x': 0.741895,
    'gravity_y': 0.669768,
    'gravity_z': -0.031672,
    'rotationRate_x': 0.316738,
    'rotationRate_y': 0.77818,
    'rotationRate_z': 1.082764,
    'userAcceleration_x': 0.294894,
    'userAcceleration_y': -0.184493,
    'userAcceleration_z': 0.377542
}

payload = json.dumps(data)

response = requests.post(url, data=payload)

response_json = response.json()
print(response_json)
