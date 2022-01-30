from fastapi import FastAPI
from PIL import Image
import numpy as np
from pydantic import BaseModel
from typing import Any
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

class Input(BaseModel):
    data: str


app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load('/content/base_model.pt'))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('success')


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/predict')
def get_recycle_category(data: Input):
    # received = data.dict()
    # transformed_image = data_transforms['TEST'](np.image(received['image']))

    # dataloader = DataLoader([transformed_image], batch_size=1)
    # with torch.no_grad():
    #   inputs = next(iter(dataloader)).to(device)

    #   outputs = model(inputs)
    #   _, preds = torch.max(outputs, 1)
    #   pred_name = class_names[preds[0]]

    return {'prediction': 'out'}