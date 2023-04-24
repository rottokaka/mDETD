from flask import Flask, render_template, request, Response
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
# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

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
    parser = argparse.ArgumentParser('ViT Deformable DETR training and evaluation script', parents=[get_args_parser("debug", version)])
    args = parser.parse_args()

    model, _, _ = build_model(args)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    return model

def stream():
    model = get_model('3')
    
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
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    img_file = request.files['img_file']
    img_path = "./mDETD/images/"+img_file.filename
    img_file.save(img_path)

    im = Image.open(img_path).convert('RGB')

    # version = request.form.get('version')
    # model = get_model(request.form.get("version"))
    # im = Image.open(img_path).convert('RGB')
    # frame = np.array(im)
    # frame = frame[:, :, ::-1].copy()

    # # mean-std normalize the input image (batch-size: 1)
    # img = transform(im).unsqueeze(0)

    # # propagate through the model
    # with torch.no_grad():
    #     outputs = model([img.squeeze().to('cuda:0')])

    # # # keep only predictions with 0.7+ confidence
    # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # keep = probas.max(-1).values >= min(torch.topk(probas.max(-1).values,3).values).item()
    # # keep = probas.max(-1).values > 0.7

    # # # convert boxes from [0; 1] to image scales
    # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].to('cpu'), im.size)

    # text = get_results(im, probas[keep], bboxes_scaled)

    # frame = draw_results(frame, probas[keep], bboxes_scaled)
        
    # output_image_path = os.path.join("./mDETD/images/", img_file.filename, '.jpg')
    # cv2.imwrite(output_image_path, frame)

    # torch.cuda.empty_cache()
    return render_template('index.html', prediction=os.getcwd())

@app.route('/video')
def video():
    return
    # return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=3000, debug=True)