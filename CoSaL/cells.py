from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 
import numpy as np
import tensorflow as tf
import sys 
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops 
from tensorflow.python.ops import rnn_cell_impl 

import tensorflow.contrib.slim as slim
try:
	RNNCell = rnn_cell_impl._RNNCell 
except:
	RNNCell = rnn_cell_impl.RNNCell 

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs

try:
	from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear 
except:
	from tensorflow.python.ops.rnn_cell_impl import _linear

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh 

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"
import tensorflow as tf
resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim
import collections
import contextlib
import hashlib
import math
import numbers
from tensorflow.python.ops import init_ops

@contextlib.contextmanager
def _checked_scope(cell, scope, reuse=None, **kwargs):
  if reuse is not None:
    kwargs["reuse"] = reuse
  with vs.variable_scope(scope, **kwargs) as checking_scope:
    scope_name = checking_scope.name
    if hasattr(cell, "_scope"):
      cell_scope = cell._scope  # pylint: disable=protected-access
      if cell_scope.name != checking_scope.name:
        raise ValueError(
            "Attempt to reuse RNNCell %s with a different variable scope than "
            "its first use.  First use of cell was with scope '%s', this "
            "attempt is with scope '%s'.  Please create a new instance of the "
            "cell if you would like it to use a different set of weights.  "
            "If before you were using: MultiRNNCell([%s(...)] * num_layers), "
            "change to: MultiRNNCell([%s(...) for _ in range(num_layers)]).  "
            "If before you were using the same cell instance as both the "
            "forward and reverse cell of a bidirectional RNN, simply create "
            "two instances (one for forward, one for reverse).  "
            "In May 2017, we will start transitioning this cell's behavior "
            "to use existing stored weights, if any, when it is called "
            "with scope=None (which can lead to silent model degradation, so "
            "this error will remain until then.)"
            % (cell, cell_scope.name, scope_name, type(cell).__name__,
               type(cell).__name__))
    else:
      weights_found = False
      try:
        with vs.variable_scope(checking_scope, reuse=True):
          vs.get_variable(_WEIGHTS_VARIABLE_NAME)
        weights_found = True
      except ValueError:
        pass
      if weights_found and reuse is None:
        raise ValueError(
            "Attempt to have a second RNNCell use the weights of a variable "
            "scope that already has weights: '%s'; and the cell was not "
            "constructed as %s(..., reuse=True).  "
            "To share the weights of an RNNCell, simply "
            "reuse it in your second calculation, or create a new one with "
            "the argument reuse=True." % (scope_name, type(cell).__name__))

    # Everything is OK.  Update the cell's scope and yield it.
    cell._scope = checking_scope  # pylint: disable=protected-access
    yield checking_scope
 


class BasicLSTMCell(RNNCell):
	"""Basic LSTM recurrent network cell.
	The implementation is based on: http://arxiv.org/abs/1409.2329.
	We add forget_bias (default: 1) to the biases of the forget gate in order to
	reduce the scale of forgetting in the beginning of the training.
	It does not allow cell clipping, a projection layer, and does not
	use peep-hole connections: it is the basic baseline.
	For advanced models, please use the full LSTMCell that follows.
	"""

	def __init__(self, num_units, forget_bias=1.0, input_size=None,
			   state_is_tuple=True, activation=tf.nn.tanh, reuse=None):
		"""Initialize the basic LSTM cell.
		Args:
		num_units: int, The number of units in the LSTM cell.
		forget_bias: float, The bias added to forget gates (see above).
		input_size: Deprecated and unused.
		state_is_tuple: If True, accepted and returned states are 2-tuples of
		the `c_state` and `m_state`.  If False, they are concatenated
		along the column axis.  The latter behavior will soon be deprecated.
		activation: Activation function of the inner states.
		reuse: (optional) Python boolean describing whether to reuse variables
		in an existing scope.  If not `True`, and the existing scope already has
		the given variables, an error is raised.
		""" 
		self._num_units = num_units
		self._forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self._activation = activation
		self._reuse = reuse

	@property
	def state_size(self):
		# return LSTMStateTuple(self._num_units, self._num_units)
		return (self._num_units, self._num_units) 

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		"""Long short-term memory cell (LSTM)."""
		with _checked_scope(self, scope or "basic_lstm_cell", 
			reuse=self._reuse):
			# Parameters of gates are concatenated into one multiply for efficiency.
			c, h = state 
			# i = input_gate, j = new_input, f = forget_gate, o = output_gate
			concat = _linear([inputs, h], 4 * self._num_units, True)
			i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

			new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + \
				tf.nn.sigmoid(i) * self._activation(j))

			new_h = self._activation(new_c) * tf.nn.sigmoid(o)

			new_state = (new_c, new_h) 

			return new_h, new_state


bias_ones = init_ops.constant_initializer(1.0, dtype=tf.float32)