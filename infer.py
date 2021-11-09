import torch
import os
from PIL import Image
from models.SimpleCNNModel import SimpleCNNModel

from torchvision import transforms

# using a picture to verify the inference

model = SimpleCNNModel()
name = 'cnn_momentum'
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()
experiment_path = os.path.join("experiments", name)
best_model_filepath = os.path.join(experiment_path, "model_best.pth.tar")
if os.path.exists(best_model_filepath):
    print(f"Loading best model from {best_model_filepath}...")
    checkpoint = torch.load(best_model_filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("load model success")
model.eval()


img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
image = img_transforms(Image.open('./test_image.jpg')).unsqueeze(0)
output = model(image.cuda())
predict = output.argmax(dim = 1,keepdim=True)
cls = predict.item()
class_dic={0:"Tshirt",1:"trouser",2:"pullover",3:"dress",4:"coat",5:"sandal",6:"shirt",7:"sneaker",8:"bag",9:"ankle boot"}
print("result:",class_dic[cls])