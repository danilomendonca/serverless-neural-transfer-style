import cv2
import base64
import numpy as np

def main(args):
    global cache
    if 'cache' not in globals():
        net = cv2.dnn.readNetFromTorch("./models/instance_norm/mosaic.t7")
        cache = net
    else:
        net = cache

    img = cv2.imdecode(np.fromstring(base64.b64decode(args["image"]), dtype=np.uint8), 1)
    h, w = img.shape[:2]
    width = 400
    if w <= width:
        width = w
        dim = (w, h)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(img, 1.0, dim, (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.reshape((3, out.shape[2], out.shape[3]))
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.680
    out = out.transpose(1,2,0)
    out = cv2.convertScaleAbs(out)
    # retval, buffer = cv2.imencode('.png', out)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    retval, buffer = cv2.imencode('.jpg', out, encode_param)
    return {"statusCode":200,
            "headers":{"Content-Type":"application/json"},
            "body":{"stylizedImage":base64.b64encode(buffer)},
            "cache": cache}

# if __name__ == '__main__':
#    main({})
