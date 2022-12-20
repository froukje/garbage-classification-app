import bentoml
import numpy as np
from bentoml.io import Image as bentoImage
from bentoml.io import NumpyNdarray
from PIL import Image
import torch
from torchvision import transforms

runner = bentoml.pytorch.get("garbage_classification:latest").to_runner()
#service = bentoml.Service(name="garbage_classification_service", runners=[runner])

#@service.api(input=bentoImage(), output=NumpyNdarray(dtype="float32"))
#async 
def predict(f: bentoImage()) -> "np.ndarray":
    print(f)
    
    print(type(f))
    img_size = 224
    transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((img_size, img_size)),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
    
    input_ = transform(f)
    input_ = input_.unsqueeze_(0)
    runner.init_local()
    y_pred = runner.run(input_)#runner.async_run(input_)
    result = {
            "class_probabilites": y_pred,
            "predicted class": torch.argmax(y_pred).item()
            }
    print(result)
    #return await result
    return result

# for local testing
img = "/home/frauke/pytorch-introduction/data/garbage/extra/img1.jpeg"
img = Image.open(img, mode='r')
predict(img)

