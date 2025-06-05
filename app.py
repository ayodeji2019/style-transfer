from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import garment_style_transfer

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        content_file = request.files["content"]
        style_file = request.files["style"]

        content_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(content_file.filename))
        style_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(style_file.filename))
        content_file.save(content_path)
        style_file.save(style_path)

        output_path = os.path.join(app.config["OUTPUT_FOLDER"], "garment_stylized_output3.jpg")
        garment_style_transfer.process_style_transfer(content_path, style_path, output_path)

        return render_template("index.html", result="output/garment_stylized_output3.jpg")
    return render_template("index.html", result=None)

@app.route('/output/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)