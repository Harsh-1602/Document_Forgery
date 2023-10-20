from flask import Flask,request,render_template
import pickle
import os 
from werkzeug.utils import secure_filename
import ultralytics
from ultralytics import YOLO
from PIL import Image

app=Flask(__name__)



# model = pickle.load(open('Model_deploy/models/Whitener.pkl','rb'))
model= YOLO("Model_deploy/models/best.pt")

upload_folder=os.path.join('Model_deploy','static','uploads')
predicted_folder=os.path.join('Model_deploy','static','Predicted')
app.config['PREDICTED']=predicted_folder
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
    # n_img=img.resize((640,640))
    # img=n_img

    imag=model.predict(img)
    img_arr=imag[0].plot()
    pred=Image.fromarray(img_arr[...,::-1])

    pred.save(os.path.join(app.config['PREDICTED'], filename))

    # print(type(pred))
    img1 = os.path.join(app.config['PREDICTED'], filename)

    return render_template("final.html",img=img)




if(__name__=="__main__"):
    app.run(debug=True, port=8001)
