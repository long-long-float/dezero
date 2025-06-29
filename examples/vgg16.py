if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from PIL import Image
import dezero
from dezero.models import VGG16


# url = 'https://placecats.com/neo/300/300'
url = 'https://placecats.com/louie/300/200'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)

x = VGG16.preprocess(img)
x = x[np.newaxis]

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
    model.export(x, to_dir='vgg16')

predict_id = np.argmax(y.data)
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])