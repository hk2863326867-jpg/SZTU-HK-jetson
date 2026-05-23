import tensorflow as tf
import inspect
tf.compat.v1.disable_v2_behavior()

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

class _Linear(object):
    """Linear layer for GRU cell."""
    def __init__(self, inputs, output_size, use_bias, bias_initializer=None, kernel_initializer=None):
        self._use_bias = use_bias
        self._output_size = output_size
        self._bias_initializer = bias_initializer or tf.compat.v1.constant_initializer(0.0)
        self._kernel_initializer = kernel_initializer
        self._variables = {}

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        combined_inputs = tf.concat(inputs, axis=-1)
        input_size = combined_inputs.get_shape().as_list()[-1]
        
        with vs.variable_scope("linear"):
            if "kernel" not in self._variables:
                self._variables["kernel"] = vs.get_variable(
                    "kernel", [input_size, self._output_size],
                    initializer=self._kernel_initializer)
            if self._use_bias and "bias" not in self._variables:
                self._variables["bias"] = vs.get_variable(
                    "bias", [self._output_size],
                    initializer=self._bias_initializer)
        
        output = tf.matmul(combined_inputs, self._variables["kernel"])
        if self._use_bias:
            output = output + self._variables["bias"]
        
        return output

class VecAttGRUCell(GRUCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(num_units, reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
    
    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)
        
    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = tf.compat.v1.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h

def dynamic_rnn(cell, inputs, att_scores=None, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    Performs fully dynamic unrolling of `inputs`.

    Args:
      cell: An instance of RNNCell.
      inputs: The RNN inputs.
        If `time_major == False` (default), this must be a `Tensor` of shape:
          `[batch_size, max_time, ...]`, or a nested tuple of such
          elements.
        If `time_major == True`, this must be a `Tensor` of shape:
          `[max_time, batch_size, ...]`, or a nested tuple of such
          elements.
      att_scores: (optional) Attention scores for VecAttGRUCell.
      sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      initial_state: (optional) An initial state for the RNN.
      dtype: (optional) The data type for the initial state and expected output.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
      scope: VariableScope for the created subgraph; defaults to "rnn".

    Returns:
      A pair (outputs, state) where:
        outputs: The RNN output `Tensor`.
        state: The final state.
    """
    from tensorflow.python.ops.rnn import dynamic_rnn as tf_dynamic_rnn
    
    # 对于VecAttGRUCell，我们需要传递att_scores
    if hasattr(cell, 'call') and len(inspect.signature(cell.call).parameters) == 3:
        # 使用自定义的dynamic_rnn实现来处理att_scores
        return _dynamic_rnn_with_att_scores(cell, inputs, att_scores, sequence_length, initial_state, dtype, parallel_iterations, swap_memory, time_major, scope)
    else:
        # 对于普通RNNCell，使用默认的dynamic_rnn
        return tf_dynamic_rnn(cell, inputs, sequence_length=sequence_length, initial_state=initial_state, dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope)

def _dynamic_rnn_with_att_scores(cell, inputs, att_scores, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None):
    """Internal implementation of Dynamic RNN with attention scores."""
    from tensorflow.python.ops import rnn
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import ops
    from tensorflow.python.framework import tensor_shape
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import control_flow_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import tensor_array_ops
    from tensorflow.python.util import nest

    if not rnn_cell_impl._like_rnncell(cell):
        raise TypeError("cell must be an instance of RNNCell")

    flat_input = nest.flatten(inputs)

    if not time_major:
        flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
        flat_input = tuple(rnn._transpose_batch_time(input_) for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = math_ops.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError(
                "sequence_length must be a vector of length batch_size, "
                "but saw shape: %s" % sequence_length.get_shape())
        sequence_length = array_ops.identity(sequence_length, name="sequence_length")

    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)
        batch_size = rnn._best_effort_input_batch_size(flat_input)

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If there is no initial_state, you must give a dtype.")
            state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = tf.stack(shape)
            return tf.debugging.assert_equal(
                x_shape, packed_shape,
                message="Expected shape for Tensor %s" % x.name)

        if sequence_length is not None:
            with ops.control_dependencies(
                [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(
                    sequence_length, name="CheckSeqLen")

        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

        (outputs, final_state) = _dynamic_rnn_loop(
            cell,
            inputs,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            att_scores=att_scores,
            sequence_length=sequence_length,
            dtype=dtype)

        if not time_major:
            outputs = nest.map_structure(rnn._transpose_batch_time, outputs)

        return (outputs, final_state)

def _dynamic_rnn_loop(cell, inputs, initial_state, parallel_iterations, swap_memory, att_scores=None, sequence_length=None, dtype=None):
    """Internal implementation of Dynamic RNN loop with attention scores."""
    from tensorflow.python.ops import rnn
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import ops
    from tensorflow.python.framework import tensor_shape
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import control_flow_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import tensor_array_ops
    from tensorflow.python.util import nest

    state = initial_state
    assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

    state_size = cell.state_size

    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)

    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = rnn._best_effort_input_batch_size(flat_input)

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape inference,"
                " but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input in a "
                "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    def _create_zero_arrays(size):
        size = rnn._concat(batch_size, size)
        return array_ops.zeros(
            tf.stack(size), rnn._infer_state_dtype(dtype, state))

    flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                      flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def _create_ta(name, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            tensor_array_name=base_name + name)

    output_ta = tuple(_create_ta("output_%d" % i,
                               rnn._infer_state_dtype(dtype, state))
                    for i in range(len(flat_output_size)))
    input_ta = tuple(_create_ta("input_%d" % i, flat_input[i].dtype)
                   for i in range(len(flat_input)))

    input_ta = tuple(ta.unstack(input_)
                   for ta, input_ in zip(input_ta, flat_input))

    def _time_step(time, output_ta_t, state, att_scores=None):
        """Take a time step of the dynamic RNN."""
        input_t = tuple(ta.read(time) for ta in input_ta)
        for input_, shape in zip(input_t, inputs_got_shape):
            input_.set_shape(shape[1:])

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        if att_scores is not None:
            att_score = att_scores[:, time, :]
            call_cell = lambda: cell(input_t, state, att_score)
        else:
            call_cell = lambda: cell(input_t, state)

        if sequence_length is not None:
            (output, new_state) = rnn._rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)
        else:
            (output, new_state) = call_cell()

        output = nest.flatten(output)

        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, output))
        if att_scores is not None:
            return (time + 1, output_ta_t, new_state, att_scores)
        else:
            return (time + 1, output_ta_t, new_state)

    if att_scores is not None:
        _, output_final_ta, final_state, _ = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, output_ta, state, att_scores),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)
    else:
        _, output_final_ta, final_state = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, output_ta, state),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

    final_outputs = tuple(ta.stack() for ta in output_final_ta)

    for output, output_size in zip(final_outputs, flat_output_size):
        shape = rnn._concat(
            [const_time_steps, const_batch_size], output_size, static=True)
        output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)

    return (final_outputs, final_state)
