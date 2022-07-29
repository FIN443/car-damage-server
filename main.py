import os
from base64 import encodebytes
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import cv2

# from tensorflow.keras.utils import get_file
from models import Yolov4

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
CORS(app, expose_headers="Authorization")

uploads_dir = os.path.join(app.instance_path, "uploads")
os.makedirs(uploads_dir, exist_ok=True)


def get_response_image(img):
    binary_cv = cv2.imencode(".PNG", img)[1].tobytes()  # convert image to byte array
    encoded_img = encodebytes(binary_cv).decode("ascii")  # encode as base64
    return encoded_img


def delete_all_files(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return "Remove All File"
    else:
        return "Directory Not Found"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "image0" not in request.files:
            return jsonify({"ok": False, "message": "not exist image"})
        image_num = len(request.files)
        if image_num > 8:
            return jsonify({"ok": False, "message": "too many images"})

        cv2_img = []
        # Save images
        for i in range(image_num):
            file = request.files[f"image{i}"]
            file.save(os.path.join(uploads_dir, file.filename))
            # 1) load images
            cv2_img.append(cv2.imread(f"instance/uploads/image{i}.png")[:, :, ::-1])

        # 2) load model
        # url = "https://files.mybox.naver.com/.fileLink/nKo0AOxUtGjnYEsBihv62Gh3UP2e%2FUYNjqj7QXhgiN2xf8vtjPTV%2BG4eLBUZ0CEA15t4kk%2BKQqzu%2B9mmm6UpTAI%3D/weights.h5?authtoken=ReTsNfD9wtF1D4qqvSKg9AI%3D"
        # weights_path = get_file("weights.h5", url)
        model = Yolov4(weight_path="instance/saved_models/weights.h5", class_name_path="instance/classes.txt")

        data = []
        # Predict images
        for i in range(image_num):
            # 3) predict image
            result_img, result_bbox = model.predict_img(cv2_img[i], return_output=True, plot_img=False, show_shape=False, show_box_count=False)

            # 4) get damage categories
            kind = result_bbox["class_name"].tolist()
            kind = list(set(kind))

            # 5) encoding image
            endcoded_img = get_response_image(result_img[:, :, ::-1])
            data.append({"kind": kind, "imageBytes": endcoded_img})

        # 6) clean uploads
        delete_all_files("instance/uploads")

        return jsonify({"ok": True, "message": "done", "data": data})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
    # app.run()
