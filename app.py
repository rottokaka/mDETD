from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    img_file = request.files['img_file']
    img_path = "./images/"+img_file.filename
    img_file.save(img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)