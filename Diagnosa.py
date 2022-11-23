#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow import keras

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
model = keras.models.load_model('Model.h5')

@app.route('/recomendation', methods = ['POST'])
def predict1():
    if request.method == 'POST':
        item1 = int(request.form.get('item1'))
        item2 = int(request.form.get('item2'))
        item3 = int(request.form.get('item3'))
        item4 = int(request.form.get('item4'))
        item5 = int(request.form.get('item5'))
        item6 = int(request.form.get('item6'))
        item7 = int(request.form.get('item7'))
        item8 = int(request.form.get('item8'))
        item9 = int(request.form.get('item9'))
        item10 = int(request.form.get('item10'))
        item = {
            'Tidak mau makan': [item1],
            'Luka/lepuh pada daerah mulut' : [item2],
            'luka/lepuh pada daerah keempat kakinya' : [item3],
            'Suhu Sapi' : [item4],
            'Lama Demam' : [item5],
            'Gemetaran' : [item6],
            'hewan lebih senang berbaring' : [item7],
            'Mengalami salah satu gejala Menggeretakkan gigi/ menggosokkan mulut/ leleran mulut' : [item8],
            'terjadi penurunan produksi susu' : [item9],
            'nafas cepat' : [item10],
            }
        data = pd.DataFrame(item)
        y_pred = model.predict(data)
        y_pred = np.argmax(y_pred, axis=-1)
        y_pred.astype(int)
        return jsonify(
            {
                "status": "Success",
                "message": "Successfully making prediction",
                "data": y_pred
            }
        )
@app.route('/')
def main():
    return 'Hallow'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')