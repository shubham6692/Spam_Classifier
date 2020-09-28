import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
loaded_model=pickle.load(open('spam_detect_model.pkl', 'rb'))
cv=pickle.load(open('transform_bow.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    message =request.form['message']
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction =loaded_model.predict(vect)
    return render_template('result.html' ,prediction = my_prediction)


if __name__ == "__main__":
    app.run(debug=True)