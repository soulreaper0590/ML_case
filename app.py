import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
app = Flask(__name__)
model = pickle.load(open('rbf_clf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = []
    for x in request.form.values():
        try :
            int_features.append(int(x))
        except:
            int_features.append(float(x))
    # int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    # print(prediction[0])
    if(prediction[0]):
        return render_template('index.html', prediction_text='Personal Loan Would Be Accepted')
    else:
        return render_template('index.html', prediction_text='Personal Loan Would NOT Be Accepted')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)