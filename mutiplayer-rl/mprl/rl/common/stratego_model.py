import logging
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
from .stratego_preprocessor import STRATEGO_PREPROCESSOR
from mprl.utils import with_base_config
import json

tf = try_import_tf()
if tf is not None:
    from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

STRATEGO_MODEL = 'stratego_model'

PARTIALLY_OBSERVABLE = 'partially_observable'
FULLY_OBSERVABLE = 'fully_observable'
BOTH_OBSERVATIONS = 'both_observations'

DEFAULT_STRATEGO_MODEL_CONFIG = {
    # === Built-in options ===
    # Filter config. List of [out_channels, kernel, stride] for each filter
    "conv_filters": None,
    # Nonlinearity for built-in convnet
    "conv_activation": "relu",
    # Nonlinearity for fully connected net (tanh, relu)
    "fcnet_activation": "relu",
    # Number of hidden layers for fully connected net
    "fcnet_hiddens": [256, 256],

    # Whether layers should be shared for the value function.
    "vf_share_layers": True,

    # == LSTM ==
    # Whether to incorporate an LSTM in the model
    "use_lstm": False,
    # Max seq len for training the LSTM, defaults to 20
    "max_seq_len": 20,
    # Size of the LSTM cell
    "lstm_cell_size": 256,

    # === Options for custom models ===
    # Name of a custom preprocessor to use
    "custom_preprocessor": STRATEGO_PREPROCESSOR,
    # Name of a custom model to use
    "custom_model": STRATEGO_MODEL,
    # Extra options to pass to the custom classes
    "custom_options": {
        "mask_invalid_actions": True,
        "observation_mode": None,
        "policy_keras_model_file_path": None,
        "q_fn": False
    },
}


def print_layer(layer, msg):
    print_op = tf.print(msg, layer, summarize=200)
    with tf.control_dependencies([print_op]):
        out = tf.add(layer, layer)
    return out


class StrategoModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        model_config = with_base_config(base_config=DEFAULT_STRATEGO_MODEL_CONFIG, extra_config=model_config)
        TFModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        print(model_config)

        observation_mode = model_config['custom_options']['observation_mode']
        if observation_mode == PARTIALLY_OBSERVABLE:
            self.pi_obs_key = 'partial_observation'
            self.vf_obs_key = 'partial_observation'
        elif observation_mode == FULLY_OBSERVABLE:
            self.pi_obs_key = 'full_observation'
            self.vf_obs_key = 'full_observation'
        elif observation_mode == BOTH_OBSERVATIONS:
            self.pi_obs_key = 'partial_observation'
            self.vf_obs_key = 'full_observation'
            assert not model_config['vf_share_layers']
        else:
            assert False, "policy observation_mode must be in [PARTIALLY_OBSERVABLE, FULLY_OBSERVABLE, BOTH_OBSERVATIONS]"

        if model_config["custom_preprocessor"]:
            print(obs_space)

            self.preprocessor = ModelCatalog.get_preprocessor_for_space(observation_space=self.obs_space.original_space,
                                                                        options=model_config)
        else:
            self.preprocessor = None
            logger.warn("No custom preprocessor for StrategoModel was specified.\n"
                        "Some tree search policies may not initialize their placeholders correctly without this.")

        self.use_lstm = model_config['use_lstm']
        self.lstm_cell_size = model_config['lstm_cell_size']
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.mask_invalid_actions = model_config['custom_options']['mask_invalid_actions']

        conv_activation = get_activation_fn(model_config.get("conv_activation"))
        cnn_filters = model_config.get("conv_filters")
        fc_activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens")

        if self.use_lstm:
            state_in = [tf.keras.layers.Input(shape=(self.lstm_cell_size,), name="pi_lstm_h"),
                        tf.keras.layers.Input(shape=(self.lstm_cell_size,), name="pi_lstm_c"),
                        tf.keras.layers.Input(shape=(self.lstm_cell_size,), name="vf_lstm_h"),
                        tf.keras.layers.Input(shape=(self.lstm_cell_size,), name="vf_lstm_c")]

            seq_lens_in = tf.keras.layers.Input(shape=(), name="lstm_seq_in")
            
            self.pi_obs_inputs = tf.keras.layers.Input(
                shape=(None, *obs_space.original_space[self.pi_obs_key].shape), name="pi_observation")
    
            self.vf_obs_inputs = tf.keras.layers.Input(
                shape=(None, *obs_space.original_space[self.vf_obs_key].shape), name="vf_observation")
        
        else:
            state_in, seq_lens_in = None, None
           
            self.pi_obs_inputs = tf.keras.layers.Input(
                shape=obs_space.original_space[self.pi_obs_key].shape, name="pi_observation")

            self.vf_obs_inputs = tf.keras.layers.Input(
                shape=obs_space.original_space[self.vf_obs_key].shape, name="vf_observation")
           
              
        if cnn_filters is None:
            
            # assuming board size will always remain the same for both pi and vf networks
            if self.use_lstm:
                single_obs_input_shape = self.pi_obs_inputs.shape.as_list()[2:]
            else:
                single_obs_input_shape = self.pi_obs_inputs.shape.as_list()[1:]
            cnn_filters = _get_filter_config(single_obs_input_shape)

        def maybe_td(layer):
            if self.use_lstm:
                return tf.keras.layers.TimeDistributed(layer=layer)
            else:
                return layer

        def build_primary_layers(prefix: str, obs_in: tf.Tensor, state_in: tf.Tensor):
            # encapsulated in a function to either be called once for shared policy/vf or twice for separate policy/vf

            _last_layer = obs_in

            for i, (out_size, kernel, stride) in enumerate(cnn_filters):
                _last_layer = maybe_td(tf.keras.layers.Conv2D(
                    filters=out_size,
                    kernel_size=kernel,
                    strides=stride,
                    activation=conv_activation,
                    padding="same",
                    name="{}_conv_{}".format(prefix, i)))(_last_layer)

            _last_layer = maybe_td(tf.keras.layers.Flatten())(_last_layer)

            for i, size in enumerate(hiddens):
                _last_layer = maybe_td(tf.keras.layers.Dense(
                    size,
                    name="{}_fc_{}".format(prefix, i),
                    activation=fc_activation,
                    kernel_initializer=normc_initializer(1.0)))(_last_layer)

            if self.use_lstm:
                _last_layer, *state_out = tf.keras.layers.LSTM(
                    units=self.lstm_cell_size,
                    return_sequences=True,
                    return_state=True,
                    name="{}_lstm".format(prefix))(
                    inputs=_last_layer,
                    mask=tf.sequence_mask(seq_lens_in),
                    initial_state=state_in)
            else:
                state_out = None

            return _last_layer, state_out


        if self.use_lstm:
            pi_state_in = state_in[:2]
            vf_state_in = state_in[2:]
        else:
            pi_state_in, vf_state_in = None, None

        policy_file_path = None
        if 'policy_keras_model_file_path' in model_config['custom_options']:
            policy_file_path = model_config['custom_options']['policy_keras_model_file_path']
        if policy_file_path is not None:
            if self.use_lstm:
                raise NotImplementedError

            pi_state_out = None
            self._pi_model = load_model(filepath=policy_file_path, compile=False)
            # remove loaded input layer
            # pi_model.layers.pop(0)
            # self.pi_obs_inputs = pi_model.layers[0]

            # rename layers
            for layer in self._pi_model.layers:
                layer._name = "pi_" + layer.name
            self._pi_model.layers[-1]._name = 'pi_unmasked_logits'

            self.unmasked_logits_out = self._pi_model(self.pi_obs_inputs)

        else:
            self._pi_model = None
            pi_last_layer, pi_state_out = build_primary_layers(prefix="pi", obs_in=self.pi_obs_inputs,
                                                               state_in=pi_state_in)

            self.unmasked_logits_out = maybe_td(tf.keras.layers.Dense(
                num_outputs,
                name="pi_unmasked_logits",
                activation=None,
                kernel_initializer=normc_initializer(0.01)))(pi_last_layer)

        vf_last_layer, vf_state_out = build_primary_layers(prefix="vf", obs_in=self.vf_obs_inputs,
                                                           state_in=vf_state_in)

        if self.use_lstm:
            state_out = [*pi_state_out, *vf_state_out]
        else:
            state_out = None

        self._use_q_fn = model_config['custom_options']['q_fn']

        if self._use_q_fn:
            value_out_size = num_outputs
        else:
            value_out_size = 1

        value_out = maybe_td(tf.keras.layers.Dense(
            value_out_size,
            name="vf_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01)))(vf_last_layer)
        
        model_inputs = [self.pi_obs_inputs, self.vf_obs_inputs]
        model_outputs = [self.unmasked_logits_out, value_out]
        if self.use_lstm:
            model_inputs += [seq_lens_in, *state_in]
            model_outputs += state_out

        self.base_model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

        print(self.base_model.summary())

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        if "internal_state" in input_dict["obs"]:
            self.internal_states = input_dict["obs"]["internal_state"]

        pi_obs_inputs = input_dict["obs"][self.pi_obs_key]
        vf_obs_inputs = input_dict['obs'][self.vf_obs_key]
        if self.use_lstm:
            policy_out, self._value_out, *state_out = self.base_model([
                add_time_dimension(pi_obs_inputs, seq_lens),
                add_time_dimension(vf_obs_inputs, seq_lens),

                seq_lens,
                *state])
            policy_out = tf.reshape(policy_out, [-1, self.num_outputs])
        else:
            policy_out, self._value_out = self.base_model([pi_obs_inputs, vf_obs_inputs])
            state_out = state

        self.unmasked_policy_logits = policy_out

        if self.mask_invalid_actions:
            # set policy logits for invalid actions to zero
            self.valid_actions_masks = input_dict["obs"]["valid_actions_mask"]
            inf_mask = tf.maximum(tf.log(self.valid_actions_masks), tf.float32.min)
            self.masked_policy_logits = policy_out + inf_mask
        else:
            self.masked_policy_logits = policy_out

        return self.masked_policy_logits, state_out

    def value_function(self):
        if self._use_q_fn:
            raise NotImplementedError
        vf = tf.reshape(self._value_out, [-1])
        return vf

    def q_function(self):
        if not self._use_q_fn:
            raise NotImplementedError
        vf = tf.reshape(self._value_out, [-1, self.action_space.n])
        return vf

    def network_policy(self):
        return tf.reshape(tf.nn.softmax(logits=self.masked_policy_logits), [-1, self.action_space.n])

    @override(ModelV2)
    def get_initial_state(self):
        if self.use_lstm:
            def make_initial_state():
                return [np.zeros(self.lstm_cell_size, np.float32),
                        np.zeros(self.lstm_cell_size, np.float32)]

            if self.vf_share_layers:
                return make_initial_state()
            else:
                return [*make_initial_state(), *make_initial_state()]
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


def _get_filter_config(shape):
    shape = list(shape)
    filters_84x84 = [
        [16, [8, 8], 4],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]
    filters_42x42 = [
        [16, [4, 4], 2],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]

    filters_3x4 = [
        [32, [2, 2], 2],
        [256, [2, 2], 2]
    ]

    filters_4x4 = [
        [32, [2, 2], 2],
        [256, [2, 2], 2]
    ]

    filters_6x6 = [
        [32, [2, 2], 2],
        [256, [2, 2], 2]
    ]

    filters_10x10 = [
        [32, [2, 2], 2],
        [256, [2, 2], 2]
    ]

    if len(shape) == 3 and shape[:2] == [84, 84]:
        return filters_84x84
    elif len(shape) == 3 and shape[:2] == [42, 42]:
        return filters_42x42
    elif len(shape) == 3 and shape[:2] == [3, 4]:
        return filters_3x4
    elif len(shape) == 3 and shape[:2] == [4, 4]:
        return filters_4x4
    elif len(shape) == 3 and shape[:2] == [6, 6]:
        return filters_6x6
    elif len(shape) == 3 and shape[:2] == [10, 10]:
        return filters_10x10
    elif len(shape) == 1:
        # Don't use a cnn in this case
        return []
    else:
        raise ValueError(
            "No default configuration for obs shape {}".format(shape) +
            ", you must specify `conv_filters` manually as a model option"
            ", or add it as a default to the _get_filter_config function.")


ModelCatalog.register_custom_model(STRATEGO_MODEL, StrategoModel)

