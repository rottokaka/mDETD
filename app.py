from flask import Flask, redirect, render_template, request, Response, session, url_for
import os
import time
import argparse
import cv2

import torch
import numpy as np

from main import get_args_parser
from models import build_model
import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = "lolol"
# COCO classes
CLASSES = ['N/A', 'Blue', 'Glass', 'Head', 'Person', 'Red', 'Vest', 'White', 'Yellow']

cap = cv2.VideoCapture(0)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_ = T.Compose([
    T.Resize(480),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def get_results(pil_img, prob, boxes):
    colors = COLORS * 100
    text = ''
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        text = text+f'{CLASSES[cl]}: {p[cl]:0.2f}'
    return text

def draw_results(img, prob, boxes):
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,0,0), 2)
        cv2.putText(img, text, (int(xmin), int(ymin)),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)
    return img

def get_model(version):
    parser = argparse.ArgumentParser('ViT Deformable DETR training and evaluation script', parents=[get_args_parser("demo", version)])
    args = parser.parse_args()

    model, _, _ = build_model(args)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    return model

def stream(version):
    model = get_model(version)
    
    while True:
        ret, frame = cap.read()
            
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_pil = Image.fromarray(frame_pil)

        frame_pil = transform_(frame_pil)

        with torch.no_grad():
            outputs = model([frame_pil.squeeze().to('cuda:0')])

        # keep 3 pred with highest confident score
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values >= min(torch.topk(probas.max(-1).values,3).values).item()
        keep = probas.max(-1).values > 0.5

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].to('cpu'), frame.shape[:2][::-1])

        frame = draw_results(frame, probas[keep], bboxes_scaled)
        
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()
        
        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')



@app.route('/')
def home():
    torch.cuda.empty_cache()
    return render_template('index.html')

@app.route('/', methods=['POST'])
def home_redirect():
    session['version'] = request.form.get("version")
    if session['version'] =="0":
        session['model_name'] = "model_0"
    elif session['version'] == "1":
        session['model_name'] = "model_1"
    elif session['version'] == "2":
        session['model_name'] = "model_2"
    elif session['version'] == "3":
        session['model_name'] = "DDETR"
    elif session['version'] == "4":
        session['model_name'] = "DETReg"
    type = request.form.get("type")
    if type == "0":
        return redirect("/fimg")
    elif type == "1":
        return redirect("/fcam")


@app.route('/fimg', methods=['GET'])
def fimg():
    return render_template('img.html')

@app.route('/fimg', methods=['POST'])
def fimg_predict():
    torch.cuda.empty_cache()
    img_file = request.files['img_file']
    img_path = "./mDETD/static/"+img_file.filename
    img_file.save(img_path)

    im = Image.open(img_path).convert('RGB')

    model = get_model(session['version'])
    im = Image.open(img_path).convert('RGB')
    frame = np.array(im)
    frame = frame[:, :, ::-1].copy()

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    with torch.no_grad():
        outputs = model([img.squeeze().to('cuda:0')])

    # # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values >= 0.5 #min(torch.topk(probas.max(-1).values,3).values).item()
    # keep = probas.max(-1).values > 0.7

    # # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].to('cpu'), im.size)

    text = get_results(im, probas[keep], bboxes_scaled)

    frame = draw_results(frame, probas[keep], bboxes_scaled)
        
    output_image_path = img_path
    cv2.imwrite(output_image_path, frame)

    torch.cuda.empty_cache()
    return render_template('img.html', prediction=url_for('static', filename=img_file.filename))

@app.route('/video')
def video():
    return Response(stream(session['version']), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fcam')
def fcam():
    return render_template('cam.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)