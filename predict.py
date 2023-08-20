'''
python3 predict.py /path/to/image
python3 predict.py assets/concrete.jpg
'''
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

# Init
DIM = 128
THRESH = 0.8
img_path = sys.argv[1]
dst = 'training'
classes_path = os.path.join(dst,'classes.txt')
model_path = os.path.join(dst,'checkpoint.pt')

transform = transforms.Compose([
    transforms.Resize((DIM,DIM)),
    transforms.ToTensor()])

def split_img(img):

    cells = []
    H,W,_ = img.shape
    cols = int(W/DIM)
    rows = int(H/DIM)
    dim = (cols*DIM,rows*DIM)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    
    for r in range(rows):
        for c in range(cols):
            y_start = r*DIM
            x_start = c*DIM
            y_end = (r+1)*DIM
            x_end = (c+1)*DIM
            cell = img[y_start:y_end,x_start:x_end]
            cells.append(cell)
    
    return cells

def post_process(img,preds):
    H,W,_ = img.shape
    cols = int(W/DIM)
    rows = int(H/DIM)
    dim = (cols*DIM,rows*DIM)
    index = 0
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    for r in range(rows):
        for c in range(cols):
            color = (0,100,100)
            thickness = 1
            y_start = r*DIM
            x_start = c*DIM
            y_end = (r+1)*DIM
            x_end = (c+1)*DIM
            cord = (int((x_start+x_end)/2)-40,int((y_start+y_end)/2))
            pred = preds[index]
            pred_class = pred[0]
            pred_prob = pred[1]
            pred_prob = round(pred_prob,2)
        
            if pred_class == 'crack' and pred_prob > THRESH:
                thickness = 3
                color = (255,0,0)

            label = pred_class +" "+str(pred_prob)

            cv2.rectangle(img,(x_start,y_start),(x_end,y_end),color,thickness)
            cv2.putText(img,label,cord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            index +=1

    cv2.imwrite('res.jpg',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))

def predict(img,model,classes):

    # Read image
    im_pil = Image.fromarray(img)
    input = transform(im_pil)
    input = torch.unsqueeze(input, 0)
    input = input.to(device)

    # Predict
    output = model(input)

    # Post Process
    output = output.softmax(1)
    output = output.cpu().detach().numpy() 
    output = output.ravel()
    output_argmax = np.argmax(output)
    output_prob = output[output_argmax]
    output_class = classes[output_argmax]

    return (output_class,output_prob)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Predict on',device)

    # Load classes
    with open(classes_path,'r') as f:
        classes = f.readlines()
    classes = [x.replace('\n','') for x in classes]
    print(classes)

    # Load model
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Split to cells
    cells = split_img(img)
    print('Total cells: ', len(cells))

    # Predict
    preds = []
    for cell in cells:
        pred = predict(cell,model,classes)
        preds.append(pred)

    post_process(img,preds)
        
    print('Done!')