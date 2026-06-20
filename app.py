from flask import Flask, render_template, request, jsonify
from ragWeb import ask_question

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    question = data.get("question", "")

    result = ask_question(question)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)