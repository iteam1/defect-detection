'''
python3 infer.py /path/to/image
'''
import os
import sys
import torch
from PIL import Image
from torchvision.transforms import transforms
from torchvision import models, transforms

# Init
DIM = 128
batch_size = 16
epochs = 10
num_classes = 2
img_path = sys.argv[1]
dst = 'training'
classes_path = os.path.join(dst,'classes.txt')
model_path = os.path.join(dst,'checkpoint.pt')

transform = transforms.Compose([
    transforms.Resize((DIM,DIM)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Inference on',device)

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
    image = Image.open(img_path)
    input = transform(image)
    input = torch.unsqueeze(input, 0)
    input = input.to(device)

    # Predict
    output = model(input)

    # Post Process
    output = output.softmax(1) 
    output = torch.argmax(output)
    output = int(output)

    print(img_path,classes[output])

    print('Done!')


