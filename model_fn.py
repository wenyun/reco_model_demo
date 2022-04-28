import tensorflow as tf
from tensorflow.python.ops import math_ops, nn
import dataset


def exp_act(x, label = 'vtr'):
    max_val = 40
    if label =='cmr' :
        max_val = 300
    if label =='psr':
        max_val = 600
    return tf.minimum(tf.math.exp(x), max_val)

def exp_and_sigmoid_act(label='vtr'):
    max_value = 40.0
    if label in ['scale_vtr', 'l2r']:
        max_value = 20.0
    elif label =='cmr' :
        max_value = 300.0
    elif label =='psr':
        max_value = 600.0

    def func(x):
      return tf.minimum(tf.math.exp(x), max_value), tf.math.sigmoid(x), x
    return func


def dense_layer(inputs, units, activation, name, weight_name):
    # tf.Dense-like layer similar to that of mio-dnn
    if not isinstance(inputs, list):
        inputs = [inputs]

    assert(len(inputs) > 0)

    with tf.name_scope(name):
        i = inputs[0]
        rows = tf.shape(i)[0]
        weight = tf.get_variable(weight_name, (i.get_shape()[1]+1, units))
        bias_input = tf.fill([rows, 1], 1.0, name='bias_input')
        o = tf.matmul(tf.concat([i, bias_input], 1), weight, name=name + '_mul')

        for idx, extra_i in enumerate(inputs[1:]):
            weight = tf.get_variable(weight_name + '_extra_' + str(idx), (extra_i.get_shape()[1], units))
            o = o + tf.matmul(extra_i, weight, name=name + '_extra_mul_' + str(idx))

        if activation is not None:
            return activation(o)
        else:
            return o


def simple_dense_network(inputs, units, name, weight_name_template, act=tf.nn.relu, last_layer_no_act=False):
    output = inputs
    for i, unit in enumerate(units):
        # output = tf.layers.Dense(unit, act, name='dense_{}_{}'.format(name, i))(output)
        
        if last_layer_no_act and i == len(units) - 1:
          output = dense_layer(output, unit, None, name='dense_{}_{}'.format(name, i),
                                 weight_name=weight_name_template.format(i + 1))
        else:
          output = dense_layer(output, unit, act, name='dense_{}_{}'.format(name, i),
                                          weight_name=weight_name_template.format(i + 1))
    return output


def simple_lhuc_network(inputs, unit1, unit2, name, weight_name):
    with tf.name_scope('{}_lhuc'.format(name)):
        output = inputs
        with tf.name_scope('{}_lhuc_layer_{}'.format(name, 0)):
            output = dense_layer(output, unit1, tf.nn.relu, name='dense_{}_{}'.format(name, 0),
                                     weight_name='{}_layer1_param'.format(weight_name))
        with tf.name_scope('{}_lhuc_layer_{}'.format(name, 1)):
            output = 2.0 * dense_layer(output, unit2, tf.nn.sigmoid, name='dense_{}_{}'.format(name, 1),
                                           weight_name='{}_layer2_param'.format(weight_name))
        return output


def mmoe_layer(inputs, expert_units, name, num_experts, num_tasks, expert_act=tf.nn.relu, gate_act=tf.nn.softmax, expert_extra_inputs=None):
  if not isinstance(inputs, list):
    inputs = [inputs]
  if expert_extra_inputs is None or len(expert_extra_inputs) != num_experts:
    print("WARNING: expert_extra_inputs is None or expert_extra_inputs.size != mmoe.num_experts")
    expert_extra_inputs = None

  expert_outputs, final_outputs = [], []
  with tf.name_scope('{}_experts_network'.format(name)):
    for i in range(num_experts):
      weight_name_template = name + '_expert{}_'.format(i) + 'h{}_param'
      expert_inputs = inputs
      if expert_extra_inputs is not None:
        expert_inputs = [inputs[0], expert_extra_inputs[i]] + inputs[1:]
      expert_layer = simple_dense_network(expert_inputs, expert_units, '{}_experts'.format(name),
                                          weight_name_template, act=expert_act)
      expert_outputs.append(expert_layer)
    expert_outputs = tf.stack(expert_outputs, axis=1)  # (batch_size, num_experts, expert_units[-1])

  with tf.name_scope('{}_gates_network'.format(name)):
    for i in range(num_tasks):
      weight_name_template = name + '_task_{}_gate_'.format(i) + 'param'
      gate_layer = dense_layer(inputs, num_experts, gate_act, '{}_gates'.format(name), weight_name_template)
      gate_layer_list = tf.split(gate_layer, num_experts, axis=1)  # TODO: delete
      gate_layer = tf.reshape(gate_layer, [-1, 1, num_experts])
      if 'kwai_mmoe_2' == name and 6 == i:
        weighted_expert_output = tf.matmul(gate_layer, tf.stop_gradient(expert_outputs))
      else:
        weighted_expert_output = tf.matmul(gate_layer, expert_outputs)
      final_outputs.append(tf.reshape(weighted_expert_output, [-1, expert_units[-1]]))
  return final_outputs #(num_tasks, batch_size, expert_units[-1])

def new_multi_head_attention(query_input, action_list_input, name, nh=4, att_emb_size=32, mode='target', trt_config={}, fp16=True):
    """
    Multi-head attention
    """
    rown = tf.shape(query_input)[0]
    query_shape = query_input.get_shape().as_list()
    action_list_shape = action_list_input.get_shape().as_list()
    
    query_shape = query_input.get_shape().as_list()
    action_list_shape = action_list_input.get_shape().as_list()

    Q = tf.get_variable(name + '_q_trans_matrix', (query_shape[-1], att_emb_size * nh))  # [emb, att_emb * hn]
    K = tf.get_variable(name + '_k_trans_matrix', (action_list_shape[-1], att_emb_size * nh))
    V = tf.get_variable(name + '_v_trans_matrix', (action_list_shape[-1], att_emb_size * nh))

    querys = tf.tensordot(query_input, Q, axes=(-1, 0))  # (batch_size,sq_q,att_embedding_size*head_num)
    keys = tf.tensordot(action_list_input, K, axes=(-1, 0))
    values = tf.tensordot(action_list_input, V, axes=(-1, 0)) # (batch_size,sq_v,att_embedding_size*head_num)

    querys = tf.stack(tf.split(querys, nh, axis=2))  # (head_num,batch_size,field_sizeq,att_embedding_size)
    keys = tf.stack(tf.split(keys, nh, axis=2))      # (head_num,batch_size,field_sizek,att_embedding_size)
    values = tf.stack(tf.split(values, nh, axis=2))  # (head_num,batch_size,field_sizev,att_embedding_size)

    inner_product = tf.matmul(
      querys * (att_emb_size ** (-0.5)), 
      keys, 
      transpose_b=True)  # (head_num,batch_size,field_sizeq,field_sizek)
    normalized_att_scores = tf.nn.softmax(inner_product)   #(head_num,batch_size,field_sizeq,field_sizek)
    result = tf.matmul(normalized_att_scores, values)  # (head_num,batch_size,field_sizeq,att_embedding_sizev)
    result = tf.transpose(result,  perm=[1, 2, 0, 3]) # (batch_size,field_sizeq,hn, att_embedding_sizev)
    
    heads_output = [tf.reduce_sum(x, axis=[1, 2]) for x in tf.split(result, nh, axis=2)]
    
    mha_result = None
    if mode == 'self':
        seq_len = action_list_input.get_shape().as_list()[1]
        mha_result = tf.reshape(result, (rown, seq_len, nh * att_emb_size))
    elif mode == 'target':
        mha_result = tf.reshape(result, (rown, nh * att_emb_size))
    return mha_result, heads_output


def self_multi_head_attention(action_list_input, name='self_att'):
    self_att = new_multi_head_attention(action_list_input, action_list_input, name, nh=5, att_emb_size=32, mode='self')
    self_att += action_list_input
    return self_att

# transformer
def layer_norm(inputs, epsilon=1e-3):
    mean, variance = nn.moments(inputs, axes=-1, keep_dims=True)
    inv = math_ops.rsqrt(variance + epsilon)
    return inputs * math_ops.cast(inv, inputs.dtype) - math_ops.cast(mean * inv, inputs.dtype)


def feed_forward_network(inputs, name, use_layer_norm=False,  resnet = False, act=tf.nn.relu):
    """
        放射变化，维度不变，scale & shift
    """
    unit_size = inputs.get_shape().as_list()[-1]
    if use_layer_norm:
        inputs = layer_norm(inputs)
    output = tf.layers.dense(inputs, unit_size, act, name='{}_ffn_layer'.format(name))
    if resnet:
        return inputs + output
    else:
        return output


def transformer(query_input, action_list_input, name, nh=4, att_emb_size=32, nh_t=5, att_emb_size_t=32):
    #### self attention
    self_mha_result, _ = new_multi_head_attention(
        action_list_input, action_list_input, f'{name}_self', mode="self", nh=nh, att_emb_size=att_emb_size)
    self_mha_result_normed = layer_norm(action_list_input + self_mha_result)
    
    #### target attention
    target_mha_result, mha_splited = new_multi_head_attention(
      query_input, self_mha_result_normed, name, mode="target", nh=nh_t, att_emb_size=att_emb_size_t)
    target_norm1 = layer_norm(target_mha_result)

    return target_norm1, mha_splited




def model_fn(features, labels, mode, params):
  dense_columns = {
    'dense_%d' % i : tf.get_variable('dense_%d' % i, shape=[dataset.N_BUCKET, dataset.N_DIM])
    for i in range(dataset.DENSE_FEATURES)
  }
  
  seq_columns = {
    'seq_%d' % i : tf.get_variable('seq_%d' % i, shape=[dataset.N_ID, dataset.N_DIM])
    for i in range(dataset.SEQ_FEATURES)
  }

  id_columns = {
    'id_%d' % i : tf.get_variable('id_%d' % i, shape=[dataset.N_ID, dataset.N_DIM])
    for i in range(dataset.LABELS)
  }

  dense_input = tf.concat([
    tf.gather(dense_columns['dense_%d' % i], features['dense_%d' % i])
    for i in range(dataset.DENSE_FEATURES)
  ], axis=1)

  seq_input = [
    tf.gather(seq_columns['seq_%d' % i], features['seq_%d' % i])
    for i in range(dataset.SEQ_FEATURES)
  ]

  id_input = [
    tf.gather(id_columns['id_%d' % i], features['id_%d' % i])
    for i in range(dataset.LABELS)
  ]

  with tf.name_scope('full_rank'):
    with tf.name_scope('transformer'):
      rown = tf.shape(dense_input)[0]
      col = dense_input.get_shape()[1]
      query_input = tf.stop_gradient(tf.reshape(dense_input, (rown, 1, col)))

      with tf.name_scope('long_term'):
        long_transformer_result, long_split_output = transformer(
          query_input, seq_input[0], "long_term", nh=5)
      
      with tf.name_scope('short_term'):
        short_transformer_result, short_split_output = transformer(
          query_input, seq_input[1], "short_term", nh=5)
      
      with tf.name_scope('supershort_view'):
        ss_transformer_result, ss_split_output = transformer(
          query_input, seq_input[2], "supershort_view", nh=5)
      
      with tf.name_scope('hate'):
        hate_transformer_result, hate_split_output = transformer(
          query_input, seq_input[3], "hate", nh=5)

    transformer_output_backward = tf.concat([
      long_transformer_result, short_transformer_result, ss_transformer_result, hate_transformer_result], 1)
    transformer_output_forward = tf.stop_gradient(transformer_output_backward)
    transformer_split_output_backward = [tf.concat([x, y, z, a], 1) for x,y,z,a in zip(
      long_split_output, short_split_output, ss_split_output, hate_split_output)]
    transformer_split_output_forward = [tf.stop_gradient(x) for x in transformer_split_output_backward]
    
    task_loss = []

    with tf.name_scope('mmoe'):
      tasks_num = dataset.LABELS
      mmoe_output_1 = mmoe_layer(dense_input, [511, 255], 'mmoe', 5, tasks_num, expert_extra_inputs=transformer_split_output_backward)

      for i in range(dataset.LABELS):
        with tf.name_scope('task_%i' % i):
          tower = simple_dense_network([tf.concat([transformer_output_backward, mmoe_output_1[i], id_input[i]], 1)], 
            [255, 127, 127], 'tower_%d' % i, 'tower_%d_h{}_param' % i)
          logit = dense_layer(tower, 1, tf.nn.sigmoid, 'task_%d' % i, 'task_%d_top_param' % i)
          
          task_loss.append(
            tf.losses.log_loss(features['label_%d' % i], tf.squeeze(logit, 1), reduction='weighted_sum'))
      loss = tf.reduce_sum(task_loss)

  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(
      loss,
      global_step=tf.train.get_global_step())

  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
