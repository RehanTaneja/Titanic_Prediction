from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("titanic_model.pkl")

@app.route("/",methods=["GET","POST"])
def index():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve data from the form
        passenger_id = int(request.form["passenger_id"])
        pclass = int(request.form["pclass"])
        sex = 1 if request.form["sex"].lower() == "male" else 0
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        embarked = int(request.form["embarked"])

        # Prepare the input features
        input_features = np.array([[passenger_id, pclass, sex, age, sibsp, parch, fare, embarked]])

        # Make prediction
        prediction = model.predict(input_features)[0]
        result = "Survived 🟢" if prediction == 1 else "Did Not Survive 🔴"

        return render_template("form.html", result=result)

    except Exception as e:
        return f"An error occurred: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
