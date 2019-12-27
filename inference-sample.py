import boto3
from PIL import Image
import numpy as np
from io import BytesIO
import json

# Pleae edit the following params.
image_path = 'sample.png'
sagemaker_endpoint_name = 'hosting-deployment-aaaaaaaaaaaaaa'


img = np.array(Image.open(image_path))  # image size: 32 x 32
# swap color axis because from numpy image(H x W x C) to torch image(C X H X W)
img = img.transpose((2, 0, 1))
img = np.array([img], dtype='float32')
npy_bytes = BytesIO()
np.save(npy_bytes, img)
body = npy_bytes.getvalue()

client = boto3.client('sagemaker-runtime')

response = client.invoke_endpoint(
    EndpointName=sagemaker_endpoint_name,
    Body=body,
    ContentType='application/x-npy',
    Accept='application/json'
)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
res = json.load(response['Body'])[0]
print(classes[np.argmax(np.array(res))])
