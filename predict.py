import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import threading
import os
from uuid import uuid4
from common import cosine_similarity_loss


cosine_angle_similarity_matrix = np.load('og_train_cos_theta.npy')
# Load the pre-trained model
with tf.keras.utils.custom_object_scope({'loss': cosine_similarity_loss(cosine_angle_similarity_matrix)}):
    model = tf.keras.models.load_model('./trained')

app = Flask(__name__)

# Dictionary to store batch request status and number of records processed
batch_status = {}


def process_batch(input_filepath, output_filepath, batch_request_id):
    num_records_processed = 0
    with open(input_filepath, 'r') as f_in, open(output_filepath, 'w') as f_out:
        for line in f_in:
            data = [list(map(float, line.strip().split(',')))]
            predictions = model.predict(np.array(data))
            f_out.write(','.join(map(str, predictions[0])) + '\n')
            num_records_processed += 1
            batch_status[batch_request_id]['num_records_processed'] = num_records_processed
    # os.sleep(10)
    batch_status[batch_request_id]['status'] = 'COMPLETED'


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    input_filepath = request.json['input_filepath']
    output_filepath = request.json['output_filepath']
    batch_request_id = str(uuid4())
    batch_status[batch_request_id] = {
        'status': 'IN_PROGRESS', 'num_records_processed': 0}
    thread = threading.Thread(target=process_batch, args=(
        input_filepath, output_filepath, batch_request_id))
    thread.start()
    return jsonify({"batch_request_id": batch_request_id})


@app.route('/batch_status', methods=['GET'])
def batch_status_endpoint():
    batch_request_id = request.args['batch_request_id']
    status_info = batch_status.get(batch_request_id, {})
    return jsonify(status_info)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input_embedding']
    predictions = model.predict(np.array([data]))
    return jsonify({"upsampled_embedding": predictions[0].tolist()})


if __name__ == '__main__':
    app.run(debug=True)
