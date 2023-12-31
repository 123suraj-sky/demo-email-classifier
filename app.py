# from flask import Flask, render_template, request
# import pickle

# cv = pickle.load(open("models/cv.pkl", "rb"))
# clf = pickle.load(open("models/clf.pkl", "rb"))

# app = Flask(__name__)

# # @app.route("/", methods = ["GET", "POST"])
# @app.route("/")
# def home():
#     # text = ""
#     # if request.method == "POST":
#     #     text = request.form.get('email-content')

#     # return "Hello World! 123"
#     # return render_template("index.html", text = text)
#     return render_template("index.html")


# @app.route("/predict", methods = ["POST"])
# def predict():
#     if request.method == "POST":
#         email_text = request.form.get("email-content")
#     tokenized_email = cv.transform([email_text])
#     predictions = clf.predict(tokenized_email)
#     predictions = 1 if predictions == 1 else -1
#     return render_template("index.html", predictions=predictions, email_text=email_text)


# if __name__ == "__main__":
#     app.run(debug=True)
#     # debug = True allows to change the content without reloading it






from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)  # Get data posted as a json
    email = data['content']
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    # If the email is spam prediction should be 1
    prediction = 1 if prediction == 1 else -1
    return jsonify({'prediction': prediction, 'email': email})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)