# embedding_upsampling_service

## Setup
```
python3 -m venv .venv
in the venv
pip install -r requirements.txt
```

## Train
python train.py

## Prediction service
python predict.py

### Real-time Prediction
```
curl -X POST -H "Content-Type: application/json" -d '{"input_embedding": [YOUR_EMBEDDING_VALUES]}' http://127.0.0.1:5000/predict
```
Replace YOUR_EMBEDDING_VALUES with your embedding values.

### Batch Prediction
```
curl -X POST -H "Content-Type: application/json" -d '{"input_filepath": "path_to_input_file.txt", "output_filepath": "path_to_output_file.txt"}' http://127.0.0.1:5000/batch_predict
```
Replace path_to_input_file.txt with the path to your input file and path_to_output_file.txt with the desired path for the output file.

### Batch Status
```
curl -X GET "http://127.0.0.1:5000/batch_status?batch_request_id=YOUR_BATCH_REQUEST_ID"
```
Replace YOUR_BATCH_REQUEST_ID with the unique identifier you received from the batch request.

### Input File Format
The input file for batch prediction should contain one embedding per row, with values separated by commas. Here's an example format:
```
1.0,2.0,3.0,...,32.0
32.0,31.0,...,1.0
...
```

