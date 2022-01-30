from fastapi import FastAPI
from PIL import Image
import numpy as np
from pydantic import BaseModel
from typing import Any
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import json

from PIL import Image

class Input(BaseModel):
    image: Any

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    global data_transforms
    global device
    global class_names

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load('base_model.pt'))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('success')

    data_transforms = {
            'TRAIN': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'TEST': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }

    f = open('class_names.json')
    class_names = json.load(f)
    f.close()


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/predict')
async def get_recycle_category(data: Input):
    received = data.dict()
    image = Image.fromarray(np.array(received['image']).astype(np.uint8))
    transformed_image = data_transforms['TEST'](image)

    dataloader = DataLoader([transformed_image], batch_size=1)
    with torch.no_grad():
      inputs = next(iter(dataloader)).to(device)

      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      pred_name = class_names[preds[0]]

    return {'prediction': pred_name}
