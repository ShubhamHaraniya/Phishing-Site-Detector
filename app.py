from flask import Flask, request, render_template
from feature import FeatureExtraction
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

gbc = joblib.load('model//model_main.pkl')

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred = gbc.predict(x)[0]
        
        #1 is safe
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('result.html', xx=round(y_pro_non_phishing,2), url=url)
    return render_template("index.html", xx =-1)

if __name__ == "__main__":
    app.run(debug=True)
