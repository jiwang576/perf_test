# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import csv
import os
import json
import numpy
import pickle
import StringIO
import sys
import signal
import time
import traceback

import flask

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, 'model.pkl'), 'r') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, inputs):
        """For the input, do the predictions and return them."""
        sklearn_predictor = cls.get_model()

        try:
          return sklearn_predictor.predict(inputs)
        except Exception as e:
          print("Exception during predicting with sklearn model.")
          print(e)


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

def _read_csv_as_float(input_path):
  """Parses each row in the csv file into a list of numbers.

  This is needed for sklearn and xgboost models that expect a list of
  numbers ([1,2,3,4]) as input, unlike the Tensorflow Model which accepts
  a string (e.g "1,2,3,4").

  Args:
    input_path: Path to the input csv file.

  Returns:
    A list of instances, each a list of numbers.
  """
  instances = []
  with open(input_path) as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

    for row in reader:
      # The first item in each row is the label which we need to discard.
      instances.append(row[1:])
  return instances

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """

    start_time = time.clock()

    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        f = open("temp.csv", "wb")
        f.write(data)
        f.close()
        data = _read_csv_as_float("temp.csv")
    elif flask.request.content_type == 'text/json':
        data = flask.request.data.get('instances')

    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(len(data)))

    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    print(type(predictions))
    result = {}
    result['predictions'] = numpy.ndarray.tolist(predictions)

    end_time = time.clock()

    resp = flask.Response(response=json.dumps(result), status=200, mimetype='text/csv')
    resp.headers['python-latency'] = end_time - start_time
    return resp
