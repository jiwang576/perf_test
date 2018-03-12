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
import timeit
import traceback
from contextlib import contextmanager

import flask

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

MICRO = 1000000
MILLI = 1000


class Timer(object):
  """Context manager for timing code blocks.

  The object is intended to be used solely as a context manager and not
  as a general purpose object.

  The timer starts when __enter__ is invoked on the context manager
  and stopped when __exit__ is invoked. After __exit__ is called,
  the duration properties report the amount of time between
  __enter__ and __exit__ and thus do not change. However, if any of the
  duration properties are called between the call to __enter__ and __exit__,
  then they will return the "live" value of the timer.

  If the same Timer object is re-used in multiple with statements, the values
  reported will reflect the latest call. Do not use the same Timer object in
  nested with blocks with the same Timer context manager.

  Example usage:

    with Timer() as timer:
      foo()
    print(timer.duration_secs)
  """

  def __init__(self, timer_fn=None):
    self.start = None
    self.end = None
    self._get_time = timer_fn or timeit.default_timer

  def __enter__(self):
    self.end = None
    self.start = self._get_time()
    return self

  def __exit__(self, exc_type, value, traceback):
    self.end = self._get_time()
    return False

  @property
  def seconds(self):
    now = self._get_time()
    return (self.end or now) - (self.start or now)

  @property
  def microseconds(self):
    return int(MICRO * self.seconds)

  @property
  def milliseconds(self):
    return int(MILLI * self.seconds)


class Stats(dict):
  """An object for tracking stats.

  This class is dict-like, so stats are accessed/stored like so:

    stats = Stats()
    stats["count"] = 1
    stats["foo"] = "bar"

  This class also facilitates collecting timing information via the
  context manager obtained using the "time" method. Reported timings
  are in microseconds.

  Example usage:

    with stats.time("foo_time"):
      foo()
    print(stats["foo_time"])
  """

  @contextmanager
  def time(self, name, timer_fn=None):
    with Timer(timer_fn) as timer:
      yield timer
    self[name] = timer.microseconds

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

    stats = Stats()
    prediction_start_time = time.time()
    stats['prediction-start-time'] = prediction_start_time * MILLI
    stats['prediction-server-start-time'] = prediction_start_time * MILLI

    with stats.time('prediction-total-time'):
      data = None

      with stats.time('prediction-loads-time'):
        if flask.request.content_type == 'text/csv':
            data = flask.request.data.decode('utf-8')
            f = open("temp.csv", "wb")
            f.write(data)
            f.close()
            data = _read_csv_as_float("temp.csv")
        elif flask.request.content_type == 'text/json':
          # not implemented
            data = flask.request.data.get('instances')

        else:
            return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

      print('Invoked with {} records'.format(len(data)))

      # Do the prediction
      with stats.time('prediction-predict-time'):
        predictions = ScoringService.predict(data)

      # Convert from numpy array to json response body
      result = {}
      result['predictions'] = numpy.ndarray.tolist(predictions)
      result.update(stats)

    resp = flask.Response(response=json.dumps(result), status=200, mimetype='text/csv')
    resp.headers.extend(stats)
    return resp
