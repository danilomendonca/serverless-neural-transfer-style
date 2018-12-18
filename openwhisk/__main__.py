import os
from cv2 import dnn
import cv2
from PIL import Image
import time
import base64
import io
import json
import numpy as np

def response_message(statusCode, jsonContent):
    return {"statusCode": statusCode,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(jsonContent)}

def main(args):
    # payload = json.loads(args["body"])
    # image = Image.open(io.BytesIO(base64.b64decode(payload["image"])))
    img = cv2.imread("./testImages/chicago.jpg")
    # style = payload["style"]
    style = 0

    # get models
    model_path = './models/instance_norm/'
    models = []
    for f in sorted(os.listdir(model_path)):
        if f.endswith('.t7'):
            models.append(f)
    
    # load user chosen model
    model_loaded_i = 0
    if style in range(0, len(models)):
       model_loaded_i = style
    model_to_load = model_path  + models[model_loaded_i]
    net = dnn.readNetFromTorch(model_to_load)
    
    # resize image if necessary
    h, w = img.shape[:2]
    width = 700
    if w <= width:
        width = w
        dim = (w, h)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    
    # Feed the neural net
    blob = dnn.blobFromImage(img, 1.0, dim,
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)

    # Run neural network
    out = net.forward()

    # Add back in the mean subtraction and then swap the channel ordering
    out = out.reshape((3, out.shape[2], out.shape[3]))
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.680
    out = out.transpose(1,2,0)

    # convert image from BGR to RGB and then to uint8 without strange artifacts
    out = cv2.convertScaleAbs(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    img = Image.fromarray(out)

    # base 64 encode response
    result_buffer = io.BytesIO()
    img.save(result_buffer, format='PNG')
    img = base64.b64encode(result_buffer.getvalue())
    return response_message(200, {'stylizedImage': img})

# if __name__ == '__main__':
#    main({})
