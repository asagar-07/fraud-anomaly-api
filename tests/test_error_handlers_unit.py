from fastapi import FastAPI
from fastapi.testclient import TestClient
from fraud.api.errors import value_error_handler, unhandled_exception_handler

def test_value_error_handler_returns_422_and_payload_shape():
    app = FastAPI()
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.get("/boom")
    def boom():
        raise ValueError("Bad input")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom")

    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "ValidationError"
    assert "Bad input" in body["error"]["message"]

def test_unhandled_exception_handler_returns_500_and_generic_message():
    app = FastAPI()
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.get("/crash")
    def crash():
        raise RuntimeError("Kaboom")

    client = TestClient(app, raise_server_exceptions= False)
    resp = client.get("/crash")

    assert resp.status_code == 500
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "InternalServerError"
    # Should NOT leak internal exception text
    assert body["error"]["message"] == "Unexpected server error"