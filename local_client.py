import boto3
import pandas as pd
import pickle
import sys
import time

def get_time_for_prediction(endpoint_name, content_type, payload):
  t0 = time.time()
  response = runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                            ContentType=content_type,
                                            Body=payload)
  return (time.time() - t0) * 1000

def load_payload(file_name):
  with open(file_name, 'r') as f:
    payload = f.read().strip()
  return payload

runtime_client = boto3.client('runtime.sagemaker')

time_data = []
endpoint_name = 'BuiltInXGBoostEndpointPkl-2018-03-15-20-25-44'
content_type = 'text/csv'
payload = load_payload(sys.argv[1])

attempts = int(sys.argv[2])

for i in range(attempts):
  if i % 50 == 0:
    print('The {}-th attempt.'.format(i))
  time_data.append(get_time_for_prediction(endpoint_name, content_type, payload))

output_file = "latency-{}".format(int(time.time()))
with open(output_file, 'wb') as f:
  pickle.dump(time_data, f)

# Analyze the data.
s = pd.Series(time_data)
print(s.describe())
print('99%-tile latency: {}'.format(s.quantile([.99])))
