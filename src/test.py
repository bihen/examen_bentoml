import requests

# The URL of the login and prediction endpoints
login_url = "http://127.0.0.1:3000/login"
predict_url = "http://127.0.0.1:3000/v1/models/admission/predict"

# Données de connexion
credentials = {
    "username": "user123",
    "password": "password123"
}

# Send a POST request to the login endpoint
login_response = requests.post(
    login_url,
    headers={"Content-Type": "application/json"},
    json=credentials
)

# Check if the login was successful
if login_response.status_code == 200:
    token = login_response.json().get("token")
    print("JWT Token:", token)

    # Data to be sent to the prediction endpoint
    data = {
        "GRE": 337, 
        "TOEFL": 118,
        "University": 4,
        "SOP": 4.5,
        "LOR": 4.5,
        "CGPA": 9.65,
        "Research": 1
    }
    
    # Send a POST request to the prediction
    response = requests.post(
        predict_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        },
        json=data
    )

    print("API Response:", response.text)
else:
    print("Error duriong connection:", login_response.text)