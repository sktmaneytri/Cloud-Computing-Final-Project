from flask import Flask, render_template, request
import pickle
import numpy as np
import os

model = pickle.load(open('iri.pkl', 'rb'))
app = Flask(__name__)
port = int(os.environ.get('PORT', 3000))


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port ,debug=True)















