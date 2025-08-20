import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test if homepage loads successfully"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"chatbox" in response.data  # check chat UI exists

def test_chat_response(client):
    """Test chatbot response API"""
    response = client.post("/get_response", json={"message": "How can I open a new bank account?"})
    
    assert response.status_code == 200
    
    data = response.get_json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0

def test_chat_audio(client):
    """Test if audio file is generated in response"""
    response = client.post("/get_response", json={"message": "Hello"})
    data = response.get_json()
    
    assert "audio" in data
    assert data["audio"].endswith(".mp3")
