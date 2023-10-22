from flask import Flask,request,render_template
import pickle
import os 
from werkzeug.utils import secure_filename

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

app=Flask(__name__)



DF = pickle.load(open('models/DF_2.pkl','rb'))
OW = pickle.load(open("models/OW.pkl", "rb"))
# model= YOLO("models/best.pt")

upload_folder=os.path.join('static','uploads')
predicted_folder=os.path.join('static','Predicted')
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

    def bboxes(result, thres):
        boxes = result[0].boxes.cpu().numpy()
        # print(first)
        qu = {}
        final = []
        small = []
        conf = []
        visited = [0] * len(boxes)
        for i, box in enumerate(boxes):
            if(box.conf[0] < thres):
                continue
            if(visited[i]):
                continue
            visited[i] = 1
            r = box.xyxy[0]
            prev = r
            qu[i] = r
            # print(r)
            small = []
            con = box.conf[0]
            small.append(r)
            while qu:
                index, val = list(qu.items())[0]
                del qu[index]
                visited[index] = 1
                prev = val
                # print(val)
                for j, b in enumerate(boxes):
                    # print(j)
                    if(b.conf[0] < thres):
                        continue
                    
                    if(index == j):
                        continue
                    if(visited[j]):
                        continue

                    if( con < b.conf[0]):
                        con = b.conf[0]

                    p = b.xyxy[0]
                    dist = np.sqrt(np.square(p[0] - prev[0]) + np.square(p[1] - prev[1]))
                    # print(dist)
                    if(dist < 9):
                        qu[j] = p
                        small.append(p)
                        # print(p)
            # print(i)
                # print(small)
                # print(con)
            if(con):
                conf.append(round(con, 2))
            if(len(small)):
                final.append(small)
        
        coord = []
        for lst in final:
            min_x, min_y, max_x, max_y = 10000.0, 10000.0, -1.0, -1.0
            for lt in lst:
                if(lt[0] < min_x):
                    min_x = lt[0]
                    min_y = lt[1]
                if(lt[2] > max_x):
                    max_x = lt[2]
                    max_y = lt[3]
            coord.append([min_x, min_y, max_x, max_y])
        # print(coord)
        return coord, conf

    def plot(result_DF, result_OW, img, ):

        final_DF_coord, final_DF_conf = [], []
        final_OW_coord, final_OW_conf = [], []
        if(len(result_DF)):
            final_DF_coord, final_DF_conf = bboxes(result_DF, 0.47)
        if(len(result_OW)):
            final_OW_coord, final_OW_conf = bboxes(result_OW, 0.55)

        if(len(final_DF_conf)):
            for i, box in enumerate(final_DF_coord):
                r = [v.astype(int) for v in box]
                # print(r)                                               # print boxes
                cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 1)
                #img = cv2.rectangle(img, (r[0], r[1] - 5), (r[2], r[1]), (0, 0, 255), -1)
                text = str(round(final_DF_conf[i] * 100, 2))
                print(text)
                img = cv2.putText(img, text, (r[0], r[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (0, 0, 255), 1)
        
        if(len(final_OW_conf)):
            for i, box in enumerate(final_OW_coord):
                r = [v.astype(int) for v in box]
                # print(r)                                               # print boxes
                cv2.rectangle(img, r[:2], r[2:], (255, 0, 0), 1)
                #img = cv2.rectangle(img, (r[0], r[1] - 5), (r[2], r[1]), (0, 0, 255), -1)
                text = str(round(final_OW_conf[i] * 100, 2))
                print(text)
                img = cv2.putText(img, text, (r[0], r[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (255, 0, 0), 1)

        return img

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

    u_img = cv2.imread(os.path.join("static", "uploads", filename))
    imag_DF = DF.predict(img)
    imag_OW = OW.predict(img)
    imag = plot(imag_DF, imag_OW, u_img)
    img_arr=imag
    pred=Image.fromarray(img_arr[...,::-1])

    pred.save(os.path.join(app.config['PREDICTED'], filename))

    # print(type(pred))
    img1 = os.path.join(app.config['PREDICTED'], filename)

    return render_template("final.html",img=img1)




if(__name__=="__main__"):
    app.run(host="0.0.0.0", port=8080)
