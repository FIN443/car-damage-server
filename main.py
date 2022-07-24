import os
import io
from base64 import encodebytes
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import cv2
from PIL import Image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
CORS(app, expose_headers="Authorization")

uploads_dir = os.path.join(app.instance_path, "uploads")
os.makedirs(uploads_dir, exist_ok=True)


def get_response_image(pil_img):
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format="PNG")  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
    return encoded_img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"ok": False, "message": "not exist image"})
        file = request.files["image"]
        file.save(os.path.join(uploads_dir, file.filename))

        # 1) load model
        # model = Yolov4(weight_path=None, class_name_path="instance/classes.txt")

        # 2) load weights
        # model.yolo_model.load_weights("instance/saved_models/weights.h5")

        # 3) load image
        # cv2_img = cv2.imread("instance/uploads/image.jpg")[:, :, ::-1]

        # 4) predict image
        # result_img, result_bbox = model.predict_img(cv2_img, return_output=True)

        # 5) get damage categories

        # 6) encoding image
        result_img = cv2.imread("instance/uploads/image.jpg")[:, :, ::-1]
        real_img = Image.fromarray(result_img)
        endcoded_img = get_response_image(real_img)
        return jsonify({"ok": True, "message": "done", "data": {"kind": "damage", "imageBytes": endcoded_img}})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
    # app.run()
