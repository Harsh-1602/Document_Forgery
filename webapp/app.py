from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save(f'{app.config["UPLOAD_FOLDER"]}/uploaded_file.jpg')
    return 'File uploaded successfully!'

@app.route('/analyze', methods=['POST'])
def analyze():
    # Add code to call your machine learning model here
    # Determine if forgery was detected
    result = "Forgery detected!"  # Replace with your actual result
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
