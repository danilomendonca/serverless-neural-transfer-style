import json
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import base64
import io


def load_graph(frozen_graph_filename):
    # read protobuf pb into graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import the graph_def into a new Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


tf_nn_num_styles = 26
image_sizes = [600, 700, 800]
# 1024 richiede 512

'''
bike.jpg
bird.jpg
boat.jpg
chicago.jpg
gate.jpg
stata.jpg
tower.jpg
'''


def response_message(statusCode, jsonContent):
    return {"statusCode": statusCode,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(jsonContent)}


def main(args):
    # payload = json.loads(event["body"])

    # image = Image.open(io.BytesIO(base64.b64decode(payload["image"])))
    image = Image.open("./testImages/bird.jpg")
    # image.show()

    # selected_size = payload["size"]
    selected_size = 700
    # selected_style = payload["style"]
    selected_style = 19

    graph = load_graph("./neuralNetwork/stylize_quantized.pb")
    input_image_tensor = tf.Tensor(graph.get_operation_by_name('input'), 0, tf.float32)
    input_image_tensor.set_shape([1, selected_size, selected_size, 3])
    input_style_tensor = tf.Tensor(graph.get_operation_by_name('style_num'), 0, tf.float32)
    input_style_tensor.set_shape([tf_nn_num_styles])
    output_tensor = graph.get_tensor_by_name("transformer/expand/conv3/conv/Sigmoid:0")

    style_vals = np.empty(shape=tf_nn_num_styles, dtype=np.float32)
    style_vals.fill(0)
    if selected_style in range(0, tf_nn_num_styles):
        style_vals[selected_style] = 1;
    else:
        # select Van Gogh as fallback
        style_vals[19] = 1;

    original_size = image.size
    transformed_size = selected_size, selected_size
    image = image.resize(transformed_size)

    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    start_millis = int(round(time.time() * 1000))
    with tf.Session(graph=graph) as sess:

        output = sess.run(output_tensor,
                          feed_dict={input_image_tensor: image,
                                     input_style_tensor: style_vals
                                     }
                          )
        output_squeezed = np.array(np.squeeze(output, axis=0) * 255, dtype=np.uint8)
        image = Image.fromarray(output_squeezed).resize(original_size)
        # image = Image.fromarray(output_squeezed)
        # image.show()
        print("Image stylized successfully !!!")
        millis = int(round(time.time() * 1000)) - start_millis
        print("TF Computation time: {} milliseconds".format(millis))

        result_buffer = io.BytesIO()
        image.save(result_buffer, format='PNG')
        encoded_result = base64.b64encode(result_buffer.getvalue())
        return response_message(200, {'image': encoded_result})

# if __name__ == '__main__':
#    main({})
