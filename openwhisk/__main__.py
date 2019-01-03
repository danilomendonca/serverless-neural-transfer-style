import cv2
import base64
import numpy as np

def main(args):
    # get models
    models = ["./models/instance_norm/mosaic.t7",
              "./models/instance_norm/candy.t7" ]

    # img = cv2.imread("./testImages/chicago.jpg")
    # style = 0

    # >0 Return a 3-channel color image.
    img = cv2.imdecode(np.fromstring(base64.b64decode(args["image"]), dtype=np.uint8),1)
    style = args["style"]

    # load user chosen model
    if style >= len(models):
	style = 0
    net = cv2.dnn.readNetFromTorch(models[style])

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
    blob = cv2.dnn.blobFromImage(img, 1.0, dim,
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)

    # Run neural network
    out = net.forward()

    # Add back in the mean subtraction
    out = out.reshape((3, out.shape[2], out.shape[3]))
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.680
    out = out.transpose(1,2,0)

    # convert image from BGR to RGB and then to uint8 without strange artifacts
    out = cv2.convertScaleAbs(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    # https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imencode
    retval, buffer = cv2.imencode('.png', out)
    return {"statusCode":200,"headers":{"Content-Type":"application/json"},"body":{"stylizedImage":base64.b64encode(buffer)}}

# if __name__ == '__main__':
#    main({})
