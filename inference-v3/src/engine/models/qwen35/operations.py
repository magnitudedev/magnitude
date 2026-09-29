"""Associate architecture formulas with public ops physical bodies."""

import ops

from . import equations

dense_feedforward = ops.operation(equations.dense_feedforward)(ops.operation_bodies.dense_feedforward)
routed_feedforward = ops.operation(equations.routed_feedforward)(ops.operation_bodies.routed_feedforward)
attention_state = ops.operation(equations._attention_state)(ops.operation_bodies.attention_state)
attention_mixer = ops.operation(equations.attention_mixer)(ops.operation_bodies.attention_mixer)
recurrent_state = ops.operation(equations._recurrent_state)(ops.operation_bodies.recurrent_state)
recurrent_mixer = ops.operation(equations.recurrent_mixer)(ops.operation_bodies.recurrent_mixer)
block = ops.operation(equations.block)(ops.operation_bodies.block)
