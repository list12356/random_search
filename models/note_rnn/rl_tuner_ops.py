# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions to support the RLTuner and NoteRNNLoader classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

# internal imports

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf


LSTM_STATE_NAME = 'lstm'

# Number of output note classes. This is a property of the dataset.
NUM_CLASSES = 38

# Default batch size.
BATCH_SIZE = 128

# Music-related constants.
INITIAL_MIDI_VALUE = 48
NUM_SPECIAL_EVENTS = 2
MIN_NOTE = 48  # Inclusive
MAX_NOTE = 84  # Exclusive
TRANSPOSE_TO_KEY = 0  # C Major
DEFAULT_QPM = 80.0

# Music theory constants used in defining reward functions.
# Note that action 2 = midi note 48.
C_MAJOR_SCALE = [2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 25, 26]
C_MAJOR_KEY = [0, 1, 2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 25, 26, 28,
               30, 31, 33, 35, 37]
C_MAJOR_TONIC = 14
A_MINOR_TONIC = 23

# The number of half-steps in musical intervals, in order of dissonance
OCTAVE = 12
FIFTH = 7
THIRD = 4
SIXTH = 9
SECOND = 2
FOURTH = 5
SEVENTH = 11
HALFSTEP = 1

# Special intervals that have unique rewards
REST_INTERVAL = -1
HOLD_INTERVAL = -1.5
REST_INTERVAL_AFTER_THIRD_OR_FIFTH = -2
HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH = -2.5
IN_KEY_THIRD = -3
IN_KEY_FIFTH = -5

# Indicate melody direction
ASCENDING = 1
DESCENDING = -1

# Indicate whether a melodic leap has been resolved or if another leap was made
LEAP_RESOLVED = 1
LEAP_DOUBLED = -1

def autocorrelate(signal, lag=1):
  """Gives the correlation coefficient for the signal's correlation with itself.

  Args:
    signal: The signal on which to compute the autocorrelation. Can be a list.
    lag: The offset at which to correlate the signal with itself. E.g. if lag
      is 1, will compute the correlation between the signal and itself 1 beat
      later.
  Returns:
    Correlation coefficient.
  """
  n = len(signal)
  x = np.asarray(signal) - np.mean(signal)
  c0 = np.var(signal)
  
  if c0 == 0:
    return 0

  return (x[lag:] * x[:n - lag]).sum() / float(n) / c0


