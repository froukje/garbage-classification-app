import os
import requests
from PIL import Image
from io import BytesIO
from skimage.io import imread
#import matplotlib.image as mpimg

#url = "http://127.0.0.1:3000/predict"
url = os.getenv('API_ENDPOINT', 'http://127.0.0.1:3000/predict')
print("API_ENDPOINT", os.getenv('API_ENDPOINT'))
print("URL", url)
#img = '/home/frauke/pytorch-introduction/data/garbage/extra/img1.jpeg'
img = imread('test.jpg')

response = requests.post(
    url, headers={"content-type": "text/plain"}, data=str(img)
)

print(response)

if response.ok:
    print("Upload completed successfully!")
    print(response.text)
else:
    print("Something went wrong!")
