from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_prediction():
    payload = {
        "district": "Kamppi",
        "size_sqm": 55.5,
        "rooms": 2,
        "construction_year": 2015,
        "floor": 4,
        "has_elevator": True,
        "has_sauna": True,
        "has_balcony": True,
        "condition": "Good",
        "monthly_maintenance_fee_eur": 280.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_eur" in data
    assert data["predicted_price_eur"] > 0