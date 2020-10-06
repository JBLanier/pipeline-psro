import logging
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
from ray.rllib.models.model import restore_original_dimensions, flatten
from mprl.utils import with_base_config
import json

tf = try_import_tf()
if tf is not None:
    from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

SAC_SPATIAL_RNN_STRATEGO_MODEL = 'sac_spatial_rnn_stratego_model'

PARTIALLY_OBSERVABLE = 'partially_observable'
FULLY_OBSERVABLE = 'fully_observable'
BOTH_OBSERVATIONS = 'both_observations'

DEFAULT_STRATEGO_MODEL_CONFIG = {
    # === Built-in options ===
    # Nonlinearity for built-in convnet
    "conv_activation": "relu",
    # == LSTM ==
    # Whether to incorporate an LSTM in the model
    "use_lstm": True,
    # Max seq len for training the LSTM, defaults to 20
    "max_seq_len": 20,

    # Name of a custom model to use
    "custom_model": SAC_SPATIAL_RNN_STRATEGO_MODEL,
    # Extra options to pass to the custom classes
    "custom_options": {
        "mask_invalid_actions": True,
        "observation_mode": None,
        "base_cnn_filters": [
            [32, [3, 3], 1],
        ],
        "base_lstm_filters": [
            [32, [3, 3], 1],
        ],
        "pi_cnn_filters": [
            [32, [3, 3], 1],
        ],
        "q_cnn_filters": [
            [32, [3, 3], 1],
        ],
        "fake_lstm": False,
    },
}


def print_layer(layer, msg):
    print_op = tf.print(msg, layer, summarize=200)
    with tf.control_dependencies([print_op]):
        out = tf.add(layer, layer)
    return out


class SACSpatialRNNStrategoModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        model_config = with_base_config(base_config=DEFAULT_STRATEGO_MODEL_CONFIG, extra_config=model_config)
        TFModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        print(model_config)

        observation_mode = model_config['custom_options']['observation_mode']
        if observation_mode == PARTIALLY_OBSERVABLE:
            self._obs_key = 'partial_observation'
        elif observation_mode == FULLY_OBSERVABLE:
            self._obs_key = 'full_observation'
        elif observation_mode == BOTH_OBSERVATIONS:
            raise NotImplementedError
        else:
            assert False, "policy observation_mode must be in [PARTIALLY_OBSERVABLE, FULLY_OBSERVABLE, BOTH_OBSERVATIONS]"

        self._action_dist_class, self._logit_dim = ModelCatalog.get_action_dist(self.action_space, model_config)

        self.use_lstm = model_config['use_lstm']
        self.fake_lstm = model_config['custom_options'].get('fake_lstm', False)

        self.mask_invalid_actions = model_config['custom_options']['mask_invalid_actions']

        conv_activation = get_activation_fn(model_config.get("conv_activation"))
        base_lstm_filters = model_config["custom_options"]['base_lstm_filters']
        base_cnn_filters = model_config["custom_options"]['base_cnn_filters']
        pi_cnn_filters = model_config["custom_options"]['pi_cnn_filters']
        q_cnn_filters = model_config["custom_options"]['q_cnn_filters']

        rows = obs_space.original_space[self._obs_key].shape[0]
        colums = obs_space.original_space[self._obs_key].shape[1]

        if self.use_lstm:
            self._lstm_state_shape = (rows, colums, base_lstm_filters[0][0])

        if self.use_lstm and not self.fake_lstm:
            self._base_model_out_shape = (rows, colums, base_lstm_filters[0][0])
        else:
            self._base_model_out_shape = (rows, colums, base_cnn_filters[-1][0])

        if self.use_lstm:
            state_in = [tf.keras.layers.Input(shape=self._lstm_state_shape, name="base_lstm_h"),
                        tf.keras.layers.Input(shape=self._lstm_state_shape, name="base_lstm_c")]
            seq_lens_in = tf.keras.layers.Input(shape=(), name="lstm_seq_in")
            
            self._obs_inputs = tf.keras.layers.Input(
                shape=(None, *obs_space.original_space[self._obs_key].shape), name="observation")
            self._base_model_out = tf.keras.layers.Input(shape=self._base_model_out_shape, name="model_out")
        else:
            state_in, seq_lens_in = None, None
            self._obs_inputs = tf.keras.layers.Input(
                shape=obs_space.original_space[self._obs_key].shape, name="observation")
            self._base_model_out = tf.keras.layers.Input(shape=self._base_model_out_shape, name="model_out")

        def maybe_td(layer):
            if self.use_lstm:
                return tf.keras.layers.TimeDistributed(layer=layer, name=f"td_{layer.name}")
            else:
                return layer

        def build_shared_base_layers(prefix: str, obs_in: tf.Tensor, state_in: tf.Tensor):
            # obs_in = tf.debugging.check_numerics(
            #     obs_in, f"nan found in obs_in", name=None)

            _last_layer = obs_in

            for i, (out_size, kernel, stride) in enumerate(base_cnn_filters):
                _last_layer = maybe_td(tf.keras.layers.Conv2D(
                    filters=out_size,
                    kernel_size=kernel,
                    strides=stride,
                    activation=conv_activation,
                    padding="same",
                    name="{}_conv_{}".format(prefix, i)))(_last_layer)
                # _last_layer = tf.debugging.check_numerics(
                #     _last_layer, f"nan found in _last_layer {i}", name=None)

            base_state_out = state_in
            if self.use_lstm and not self.fake_lstm:
                for i, (out_size, kernel, stride) in enumerate(base_lstm_filters):
                    if i > 0:
                        raise NotImplementedError("Only single lstm layers are implemented right now")

                    _last_layer, *base_state_out = tf.keras.layers.ConvLSTM2D(
                        filters=out_size,
                        kernel_size=kernel,
                        strides=stride,
                        activation=conv_activation,
                        padding="same",
                        data_format='channels_last',
                        return_sequences=True,
                        return_state=True,
                        name="{}_convlstm".format(prefix))(inputs=_last_layer, initial_state=state_in, mask=tf.sequence_mask(seq_lens_in))

            return _last_layer, base_state_out

        def build_pi_layers(input_layer):
            _last_layer = input_layer
            for i, (out_size, kernel, stride) in enumerate(pi_cnn_filters):
                _last_layer = tf.keras.layers.Conv2D(
                    filters=out_size,
                    kernel_size=kernel,
                    strides=stride,
                    activation=conv_activation,
                    padding="same",
                    name="{}_conv_{}".format('pi', i))(_last_layer)

            print(f"action space n: {action_space.n}, rows: {rows}, columns: {colums}, filters: {int(action_space.n / (rows * colums))}")

            unmasked_logits = tf.keras.layers.Conv2D(
                filters=int(action_space.n / (rows * colums)),
                kernel_size=[3, 3],
                strides=1,
                activation=None,
                padding="same",
                name="{}_conv_{}".format('pi', "unmasked_logits"))(_last_layer)
            return unmasked_logits

        def build_q_layers(input_layer, prefix):
            _last_layer = input_layer
            for i, (out_size, kernel, stride) in enumerate(q_cnn_filters):
                _last_layer = tf.keras.layers.Conv2D(
                    filters=out_size,
                    kernel_size=kernel,
                    strides=stride,
                    activation=conv_activation,
                    padding="same",
                    name="{}_conv_{}".format(prefix, i))(_last_layer)

            q_val = tf.keras.layers.Conv2D(
                filters=int(action_space.n / (rows * colums)),
                kernel_size=[3, 3],
                strides=1,
                activation=None,
                padding="same",
                name="{}_conv_{}".format(prefix, "q_out"))(_last_layer)
            return q_val

        base_model_out, state_out = build_shared_base_layers(prefix="shared_base", obs_in=self._obs_inputs, state_in=state_in)
        pi_unmasked_logits_out = build_pi_layers(input_layer=self._base_model_out)
        q1_out = build_q_layers(input_layer=self._base_model_out, prefix="q1")
        q2_out = build_q_layers(input_layer=self._base_model_out, prefix="q2")

        base_inputs = [self._obs_inputs]
        base_outputs = [base_model_out]
        if self.use_lstm:
             base_inputs += [seq_lens_in, *state_in]
             base_outputs += [*state_out]

        self._base_model = tf.keras.Model(name=f"{name}_base", inputs=base_inputs, outputs=base_outputs)

        self.pi_model = tf.keras.Model(name=f"{name}_pi_head", inputs=[self._base_model_out], outputs=[pi_unmasked_logits_out])
        self.q1_model = tf.keras.Model(name=f"{name}_q1_head", inputs=[self._base_model_out], outputs=[q1_out])
        self.q2_model = tf.keras.Model(name=f"{name}_q2_head", inputs=[self._base_model_out], outputs=[q2_out])

        print(self._base_model.summary())
        print(self.pi_model.summary())
        print(self.q1_model.summary())
        print(self.q2_model.summary())

        self.register_variables(self._base_model.variables)
        self.register_variables(self.pi_model.variables)
        self.register_variables(self.q1_model.variables)
        self.register_variables(self.q2_model.variables)

        self.log_alpha = tf.Variable(0.0, dtype=tf.float32, name="log_alpha")
        self.alpha = tf.exp(self.log_alpha)
        self.register_variables([self.log_alpha])

    def forward(self, input_dict, state, seq_lens):
        obs_inputs = input_dict["obs"][self._obs_key]
        self.valid_actions_masks = input_dict["obs"]["valid_actions_mask"]
        # self.valid_actions_masks = tf.Print(input_dict["obs"]["valid_actions_mask"], [input_dict["obs"]["valid_actions_mask"]], message="valid_act_mask: ")

        if self.use_lstm:
            obs_inputs_time_dist = add_time_dimension(obs_inputs, seq_lens)
            # obs_inputs_time_dist_check = tf.debugging.check_numerics(
            #     obs_inputs_time_dist, "nan found in obs_inputs_time_dist", name=None
            # )

            # seq_lens = tf.debugging.check_numerics(
            #     seq_lens, "nan found in seq_lens", name=None
            # )
            # state_checks = []
            # for i in range(len(state)):
            #     state_checks.append(tf.debugging.check_numerics(
            #         state[i], f"nan found in state[{i}]", name=None
            #     ))

            # with tf.control_dependencies([obs_inputs_time_dist_check, *state_checks]):
            base_model_out, *state_out = self._base_model([
                obs_inputs_time_dist,
                seq_lens,
                *state])

            # base_model_out = tf.Print(base_model_out, state_out,
            #          message="state_out: ")

            return tf.reshape(base_model_out, [-1, *self._base_model_out_shape]), state_out
        else:
            base_model_out = self._base_model([obs_inputs])
            state_out = state
            return base_model_out, state_out

        # if self.mask_invalid_actions:
        #     # set policy logits for invalid actions to zero
        #     inf_mask = tf.maximum(tf.log(self.valid_actions_masks), tf.float32.min)
        #     self.masked_policy_logits = policy_out + inf_mask
        # else:
        #     self.masked_policy_logits = policy_out
        #
        # self.masked_policy_logits = tf.reshape(self.masked_policy_logits, [-1, self.num_outputs])
        #
        # return self.masked_policy_logits, state_out

    # def value_function(self):
    #     if self._use_q_fn:
    #         raise NotImplementedError
    #     vf = tf.reshape(self._value_out, [-1])
    #     return vf
    #
    # def q_function(self):
    #     if not self._use_q_fn:
    #         raise NotImplementedError
    #     vf = tf.reshape(self._value_out, [-1, self.num_outputs])
    #     return vf
    #
    # def network_policy(self):
    #     return tf.reshape(tf.nn.softmax(logits=self.masked_policy_logits), [-1, self.action_space.n])

    # def get_soft_actor_critic_discrete_policy_output(self, logits, deterministic=False):
    #     """Return the (unscaled) output of the policy network.
    #
    #     This returns the unscaled outputs of pi(s).
    #
    #     Arguments:
    #         model_out (Tensor): obs embeddings from the model layers, of shape
    #             [BATCH_SIZE, num_outputs].
    #
    #     Returns:
    #         tensor of shape [BATCH_SIZE, action_dim] with range [-inf, inf].
    #     """
    #     # https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
    #     if deterministic:
    #         actions = tf.arg_max(input=logits, dimension=1)
    #         log_pis = None
    #     else:
    #         u = tf.random.uniform(
    #             shape=tf.shape(logits),
    #             minval=0,
    #             maxval=1,
    #             dtype=tf.dtypes.float32,
    #             seed=None,
    #             name="sac_logits_noise"
    #         )
    #         noisy_logits = logits - tf.log(-tf.log(u))
    #         actions = tf.arg_max(input=noisy_logits, dimension=1)
    #
    #         one_hot_actions = tf.one_hot(indices=actions, depth=self.action_space.n)
    #         log_pis = - tf.reduce_sum(- one_hot_actions * tf.nn.log_softmax(logits, -1), axis=-1)
    #
    #     return actions, log_pis

    def get_q_values(self, base_model_out, twin_q=False):
        """Return the Q estimates for the most recent forward pass.

        This implements Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Tensor): action values that correspond with the most
                recent batch of observations passed through forward(), of shape
                [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        if twin_q:
            q_model = self.q2_model
        else:
            q_model = self.q1_model

        q_vals = q_model([base_model_out])
        return tf.reshape(q_vals, [-1, self.num_outputs])



    def get_twin_q_values(self, base_model_out):
        """Same as get_q_values but using the twin Q net.

        This implements the twin Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Tensor): action values that correspond with the most
                recent batch of observations passed through forward(), of shape
                [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        return self.get_q_values(base_model_out, twin_q=True)

    def get_policy_logits(self, base_model_out):
        logits = self.pi_model([base_model_out])

        if self.mask_invalid_actions:
            # set policy logits for invalid actions to zero
            inf_mask = tf.maximum(tf.log(self.valid_actions_masks), tf.float32.min)
            logits = logits + inf_mask

        return tf.reshape(logits, [-1, self.num_outputs])

    def get_policy_output(self, model_out, deterministic=False):
        """Return the (unscaled) output of the policy network.

        This returns the unscaled outputs of pi(s).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_dim] with range [-inf, inf].
        """
        assert self.action_space.n == self.num_outputs
        # model_out = tf.Print(model_out, [model_out], message="This is model out: ")

        logits = self.get_policy_logits(base_model_out=model_out)
        # assert len(logits.shape) == 0
        # assert logits.shape[0] == self.action_space.n

        action_dist = self._action_dist_class(logits, None)

        if deterministic:
            actions = tf.math.argmax(logits, dimension=-1)
            log_pis_for_selected_actions = None
        else:
            actions = action_dist.sample()
            # actions = tf.Print(actions, [actions], message="This is actions: ")

            # assert False, f"\n\n\nactions shape {actions.shape}, logits shape {logits.shape}\n\n\n"
            log_pis_for_selected_actions = action_dist.logp(actions)

        return actions, log_pis_for_selected_actions

    def policy_variables(self):
        """Return the list of variables for the policy net."""

        return self.pi_model.variables + self._base_model.variables

    def main_q_variables(self):
        return self.q1_model.variables + self._base_model.variables

    def twin_q_variables(self):
        return self.q2_model.variables + self._base_model.variables

    def all_q_variables(self):

        return self.q1_model.variables + self.q2_model.variables + self._base_model.variables

    @override(ModelV2)
    def __call__(self, input_dict, state=None, seq_lens=None):
        """Call the model with the given input tensors and state.

        This is the method used by RLlib to execute the forward pass. It calls
        forward() internally after unpacking nested observation tensors.

        Custom models should override forward() instead of __call__.

        Arguments:
            input_dict (dict): dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training"
            state (list): list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens (Tensor): 1d tensor holding input sequence lengths

        Returns:
            (outputs, state): The model output tensor of size
                [BATCH, output_spec.size] or a list of tensors corresponding to
                output_spec.shape_list, and a list of state tensors of
                [BATCH, state_size_i].
        """

        restored = input_dict.copy()
        restored["obs"] = restore_original_dimensions(
            input_dict["obs"], self.obs_space, self.framework)
        if len(input_dict["obs"].shape) > 2:
            restored["obs_flat"] = flatten(input_dict["obs"], self.framework)
        else:
            restored["obs_flat"] = input_dict["obs"]
        with self.context():
            res = self.forward(restored, state or [], seq_lens)
        if ((not isinstance(res, list) and not isinstance(res, tuple))
                or len(res) != 2):
            raise ValueError(
                "forward() must return a tuple of (output, state) tensors, "
                "got {}".format(res))
        outputs, state = res

        try:
            shape = outputs.shape
        except AttributeError:
            raise ValueError("Output is not a tensor: {}".format(outputs))
        # else:
        #     if len(shape) != 2 or shape[1] != self.num_outputs:
        #         raise ValueError(
        #             "Expected output shape of [None, {}], got {}".format(
        #                 self.num_outputs, shape))
        if not isinstance(state, list):
            raise ValueError("State output is not a list: {}".format(state))

        self._last_output = outputs
        return outputs, state

    @override(ModelV2)
    def get_initial_state(self):
        if self.use_lstm:
            return [np.zeros(self._lstm_state_shape, np.float32),
                    np.zeros(self._lstm_state_shape, np.float32)]
        else:
            return []

    def save_config_to_json(self, save_file_path):
        with open(save_file_path, 'w') as fp:
            json.dump(self.model_config, fp)

        # Verify that dictionary is recoverable from json
        with open(save_file_path, 'r') as fp:
            saved = json.load(fp)
        for key, orig_val in self.model_config.items():
            assert np.all(saved[key] == orig_val)


ModelCatalog.register_custom_model(SAC_SPATIAL_RNN_STRATEGO_MODEL, SACSpatialRNNStrategoModel)

