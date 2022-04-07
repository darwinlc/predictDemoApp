import re
from flask import Flask, request, url_for, redirect, render_template
from modelService import webcall

app = Flask(__name__)

# model = load_model("deployment_28042020")

supported_sec_list = ["BTC", "CMRE", "DHT", "SBLK"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    selected_sec = int_features[0]
    current_quantity = float(int_features[1])
    current_cash = float(int_features[2])
    print(selected_sec, current_quantity, current_cash)

    if selected_sec not in supported_sec_list:
        result_str = f"Security {selected_sec} not yet supported"
    else:
        # prediction
        result_str = webcall(selected_sec, current_quantity, current_cash)
        result_str = f"Result for Security {selected_sec}\n{result_str}"

    result_str = result_str.split("\n")
    return render_template("home.html", pred=result_str)


if __name__ == "__main__":
    app.run(debug=True)
