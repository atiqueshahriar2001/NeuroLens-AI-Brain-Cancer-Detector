from io import BytesIO
from PIL import Image
import app as webapp


def test_basic_routes():
    client = webapp.app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/about").status_code == 200
    assert client.get("/history").status_code == 200
    assert client.post("/history/clear").status_code == 302


def test_invalid_upload_and_bad_content():
    client = webapp.app.test_client()
    bad_extension = client.post("/analyze", data={"image": (BytesIO(b"x"), "bad.txt")}, content_type="multipart/form-data")
    assert bad_extension.status_code == 302
    bad_image = client.post("/analyze", data={"image": (BytesIO(b"not an image"), "bad.png")}, content_type="multipart/form-data")
    assert bad_image.status_code == 302
