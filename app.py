from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('knn.mdl')


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/prediksi", methods=["POST"])
def prediction():
    inputan_user = np.fromiter(request.form.values(), dtype=float)
    output = model.predict([inputan_user])[0]
    label = ['Tidak Berpotensi Tambah Daya', 'Berpotensi Tambah Daya']
    return render_template("index.html", data=label[output])


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
