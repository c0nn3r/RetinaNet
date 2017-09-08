from torchvision.transforms import Compose, Normalize, Scale, ToTensor

from PIL import Image

img = Image.open("../../Documents/2017_07/test/21094803716_da3cea21b8_o.jpg")

ready_image = Compose([
    Scale([224, 224]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),

])


