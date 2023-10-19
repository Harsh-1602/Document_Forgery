from flask import Flask,request,render_template
import os 
from werkzeug.utils import secure_filename
app=Flask(__name__)



# model = load_model('model/model.h5')

upload_folder=os.path.join('static','uploads')
app.config['UPLOAD']=upload_folder
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods=["POST"])

def predict():
    # image=request.files['file']
    # filename=secure_filename(image.filename)
    # image.save(os.path.join(app.config['UPLOAD'],filename))
    # file=os.path.join(app.config['UPLOAD'],filename)
    file = request.files['img']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD'], filename))
    img = os.path.join(app.config['UPLOAD'], filename)
    # return render_template('image_render.html', img=img)
# # sequence=tokenizers.texts_to_sequences(str(sentence))
    # # padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    # # for it in model.predict(padded):
    # #     if(it>0.5):
    # #         output="Sarcasm"
    # #     else:
    #         output="NOT Sarcasm"
    # return render_template("index.html",prediction_text="{}".format(output))
    return render_template("final.html",img=img)




if(__name__=="__main__"):
    app.run(debug=True, port=8001)
