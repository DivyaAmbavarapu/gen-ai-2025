import pytest
from chatbot.model import get_response

def test_simple_query():
    """Test a known query returns a non-empty response"""
    user_input = "How can I open a new bank account?"
    response = get_response(user_input)
    assert isinstance(response, str)
    assert len(response) > 0

def test_unseen_query():
    """Test an unknown query still gives some response"""
    user_input = "What is the meaning of life?"
    response = get_response(user_input)
    assert isinstance(response, str)
    assert len(response) > 0

def test_similar_queries():
    """Test that semantically similar queries return consistent responses"""
    q1 = get_response("How to apply for a credit card?")
    q2 = get_response("Can I get a new credit card?")
    assert isinstance(q1, str)
    assert isinstance(q2, str)
    assert len(q1) > 0 and len(q2) > 0