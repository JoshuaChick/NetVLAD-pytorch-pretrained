from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = models.vgg16(pretrained=True)
layers = list(encoder.features.children())[:-2]  # Removes the last two layers
encoder = nn.Sequential(*layers).to(DEVICE)

net_vlad = torch.jit.load('./netvlad.pt', map_location=DEVICE)


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(DEVICE)


img_tensor1 = preprocess_image('./images/buildingDay.png')
img_tensor2 = preprocess_image('./images/buildingNight.png')


encoder.eval()
net_vlad.eval()


def get_vlad_encodings(encoder_model, net_vlad_model, img):
    encoding = encoder_model(img)
    vlad_encoding = net_vlad_model(encoding)
    return vlad_encoding


vlad1 = get_vlad_encodings(encoder, net_vlad, img_tensor1)
vlad2 = get_vlad_encodings(encoder, net_vlad, img_tensor2)

dist = torch.norm(vlad1 - vlad2, p=2)
print(f'Distance: {dist.item():.2f}')
