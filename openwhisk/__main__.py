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

tf_nn_num_styles = 26;
image_sizes = [720, 1024, 1280, 1536, 1792, 2048]

def prepare_tensorflow():
    prepared_data = []
    for i in range(0, len(image_sizes)):
        graph = load_graph("./neuralNetwork/stylize_quantized.pb")
        operation_input_image = graph.get_operation_by_name('input')
        input_image_tensor = tf.Tensor(operation_input_image, 0, tf.float32)
        input_image_tensor.set_shape([1, image_sizes[i], image_sizes[i], 3])
        operation_input_style = graph.get_operation_by_name('style_num')
        input_style_tensor = tf.Tensor(operation_input_style, 0, tf.float32)
        input_style_tensor.set_shape([tf_nn_num_styles])
        output_tensor = graph.get_tensor_by_name("transformer/expand/conv3/conv/Sigmoid:0")
        prepared_data.append((graph, input_image_tensor, input_style_tensor, output_tensor))
    return prepared_data

def response_message(statusCode, jsonContent):
    return {"statusCode": statusCode,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(jsonContent)}

graphs_data = prepare_tensorflow()
print("Done preparing Tensorflow data structures!!")

def main(args):
    #payload = json.loads(event["body"])

    # image = Image.open(io.BytesIO(base64.b64decode(payload["image"])))
    image = Image.open("./testImages/bike.jpg")

    # selected_size = payload["size"]
    selected_size = 720
    print("Request size: " + str(selected_size))
    # selected_style = payload["style"]
    selected_style = 19
    print("Request style: " + str(selected_style))

    if selected_size == image_sizes[0]:
        m_graph, m_input_image_tensor, m_input_style_tensor, m_output_tensor = graphs_data[0]
    elif selected_size == image_sizes[1]:
        m_graph, m_input_image_tensor, m_input_style_tensor, m_output_tensor = graphs_data[1]
    elif selected_size == image_sizes[2]:
        m_graph, m_input_image_tensor, m_input_style_tensor, m_output_tensor = graphs_data[2]
    elif selected_size == image_sizes[3]:
        m_graph, m_input_image_tensor, m_input_style_tensor, m_output_tensor = graphs_data[3]
    elif selected_size == image_sizes[4]:
        m_graph, m_input_image_tensor, m_input_style_tensor, m_output_tensor = graphs_data[4]
    else:
       m_graph, m_input_image_tensor, m_input_style_tensor, m_output_tensor = graphs_data[5]
       if selected_size != image_sizes[5]:
           selected_size = image_sizes[5]

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

    network_input_image = np.array(image, dtype=np.float32)
    network_input_image = np.expand_dims(network_input_image, axis=0)

    start_millis = int(round(time.time() * 1000))
    with tf.Session(graph = m_graph) as sess:

        output = sess.run(m_output_tensor,
                          feed_dict = {   m_input_image_tensor: network_input_image,
                                          m_input_style_tensor: style_vals
                                     }
                          )
        output_squeezed = np.array(np.squeeze(output, axis=0)*255, dtype=np.uint8)
        stylized_image = Image.fromarray(output_squeezed).resize(original_size)
        # stylized_image.show()
        print("Image stylized successfully !!!")
        millis = int(round(time.time() * 1000)) - start_millis
        print("TF Computation time: {} milliseconds".format(millis))

        result_buffer = io.BytesIO()
        stylized_image.save(result_buffer, format='PNG')
        result_buffer = result_buffer.getvalue()
        encoded_result = base64.b64encode(result_buffer)
        return response_message(200, {'image': encoded_result})

    return response_message(500, "Internal A3E Stylize error")
