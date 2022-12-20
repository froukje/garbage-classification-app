import bentoml
import numpy as np
from bentoml.io import Image as bentoImage
from bentoml.io import Text
from PIL import Image
import torch
from torchvision import transforms

import logging

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.addHandler(ch)
bentoml_logger.setLevel(logging.INFO)

runner = bentoml.pytorch.get("garbage_classification:latest").to_runner()
service = bentoml.Service(name="garbage_classification_service", runners=[runner])

#@service.api(input=bentoImage(), output=Text())
#async def predict(f: bentoImage()) -> "string":
@service.api(input=Text(), output=Text())
async def predict(image_str: Text()) -> "string":
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    img_size = 224

    numpy_image = np.fromstring(image_str, np.uint8)
    f = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((img_size, img_size)),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
    
    input_ = transform(f)
    input_ = input_.unsqueeze_(0)
    result = await runner.async_run(input_)
    result = result.detach().cpu().numpy()  
    print(result)
    prediction = np.argmax(result)
    pred_class_ = classes[int(prediction)] 

    return pred_class_
