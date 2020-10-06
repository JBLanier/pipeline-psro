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
from ray.rllib.agents.dqn.simple_q_model import SimpleQModel
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel

tf = try_import_tf()
if tf is not None:
    from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

SPATIAL_STRATEGO_Q_MODEL = 'spatial_stratego_simple_q_model'

PARTIALLY_OBSERVABLE = 'partially_observable'
FULLY_OBSERVABLE = 'fully_observable'
BOTH_OBSERVATIONS = 'both_observations'

DEFAULT_STRATEGO_MODEL_CONFIG = {
    # === Built-in options ===
    # Nonlinearity for built-in convnet
    "conv_activation": "relu",

    "conv_filters": [
        [64, [3, 3], 1],
        [100, [3, 3], 1],
        [100, [3, 3], 1],
    ],

    # == LSTM ==
    # Whether to incorporate an LSTM in the model
    "use_lstm": False,
    # Max seq len for training the LSTM, defaults to 20
    "max_seq_len": 20,


    # === Options for custom models ===
    # Name of a custom preprocessor to use
    "custom_preprocessor": STRATEGO_PREPROCESSOR,
    # Name of a custom model to use
    "custom_model": SPATIAL_STRATEGO_Q_MODEL,
    # Extra options to pass to the custom classes
    "custom_options": {
        "mask_invalid_actions": True,
        "observation_mode": None,
        "policy_keras_model_file_path": None,
        "final_pi_filter_amt": 64,
    },
}


def print_layer(layer, msg):
    print_op = tf.print(msg, layer, summarize=200)
    with tf.control_dependencies([print_op]):
        out = tf.add(layer, layer)
    return out


class SpatialStrategoQModel(DistributionalQModel):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, q_hiddens=None, dueling=False,
                 num_atoms=1,
                 use_noisy=False,
                 v_min=-10.0,
                 v_max=10.0,
                 sigma0=0.5,
                 parameter_noise=False):

        if q_hiddens or dueling or num_atoms != 1 or use_noisy:
            raise NotImplementedError

        model_config = with_base_config(base_config=DEFAULT_STRATEGO_MODEL_CONFIG, extra_config=model_config)
        TFModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        print(model_config)

        observation_mode = model_config['custom_options']['observation_mode']
        if observation_mode == PARTIALLY_OBSERVABLE:
            self.vf_obs_key = 'partial_observation'
        elif observation_mode == FULLY_OBSERVABLE:
            self.vf_obs_key = 'full_observation'
        elif observation_mode == BOTH_OBSERVATIONS:
            raise ValueError(
                f"Using {BOTH_OBSERVATIONS} format doesn't make sense for a Q-network, there's no policy, just a Q-function")

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
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.mask_invalid_actions = model_config['custom_options']['mask_invalid_actions']

        conv_activation = get_activation_fn(model_config.get("conv_activation"))
        lstm_filters = model_config["custom_options"]['lstm_filters']
        cnn_filters = model_config.get("conv_filters")
        final_pi_filter_amt = model_config["custom_options"]["final_pi_filter_amt"]

        rows = obs_space.original_space[self.vf_obs_key].shape[0]
        colums = obs_space.original_space[self.vf_obs_key].shape[1]

        if self.use_lstm:
            self._lstm_state_shape = (rows, colums, lstm_filters[0][0])
            # self._lstm_state_shape = (64,)

        if self.use_lstm:
            state_in = [
                        tf.keras.layers.Input(shape=self._lstm_state_shape, name="vf_lstm_h"),
                        tf.keras.layers.Input(shape=self._lstm_state_shape, name="vf_lstm_c")]

            seq_lens_in = tf.keras.layers.Input(shape=(), name="lstm_seq_in")


            self.vf_obs_inputs = tf.keras.layers.Input(
                shape=(None, *obs_space.original_space[self.vf_obs_key].shape), name="vf_observation")
        
        else:
            state_in, seq_lens_in = None, None

            self.vf_obs_inputs = tf.keras.layers.Input(
                shape=obs_space.original_space[self.vf_obs_key].shape, name="vf_observation")
           
              
        # if pi_cnn_filters is None:
        #     assert False
        #     # assuming board size will always remain the same for both pi and vf networks
        #     if self.use_lstm:
        #         single_obs_input_shape = self.pi_obs_inputs.shape.as_list()[2:]
        #     else:
        #         single_obs_input_shape = self.pi_obs_inputs.shape.as_list()[1:]
        #     pi_cnn_filters = _get_filter_config(single_obs_input_shape)
        #
        # if v_cnn_filters is None:
        #     assert False
        #     # assuming board size will always remain the same for both pi and vf networks
        #     if self.use_lstm:
        #         single_obs_input_shape = self.pi_obs_inputs.shape.as_list()[2:]
        #     else:
        #         single_obs_input_shape = self.pi_obs_inputs.shape.as_list()[1:]
        #     v_cnn_filters = _get_filter_config(single_obs_input_shape)

        def maybe_td(layer):
            if self.use_lstm:
                return tf.keras.layers.TimeDistributed(layer=layer, name=f"td_{layer.name}")
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

                if parameter_noise:
                    # assuming inputs shape (batch_size x w x h x channel)
                    _last_layer = maybe_td(tf.keras.layers.LayerNormalization(
                        axis=(1, 2),
                        name=f"{prefix}_LayerNorm_{i}"
                    ))(_last_layer)


            state_out = state_in
            if self.use_lstm:
                for i, (out_size, kernel, stride) in enumerate(lstm_filters):
                    if i > 0:
                        raise NotImplementedError("Only single lstm layers are implemented right now")

                    _last_layer, *state_out = tf.keras.layers.ConvLSTM2D(
                        filters=out_size,
                        kernel_size=kernel,
                        strides=stride,
                        activation=conv_activation,
                        padding="same",
                        return_sequences=True,
                        return_state=True,
                        name="{}_convlstm".format(prefix))(
                        inputs=_last_layer,
                        mask=tf.sequence_mask(seq_lens_in),
                        initial_state=state_in)
                    raise NotImplementedError("havent checked lstms for q model"
                                              "")
            return _last_layer, state_out

        if self.use_lstm:
            vf_state_in = state_in[2:]
        else:
            pi_state_in, vf_state_in = None, None

        vf_last_layer, vf_state_out = build_primary_layers(prefix="vf", obs_in=self.vf_obs_inputs, state_in=vf_state_in)

        if self.use_lstm:
            state_out = vf_state_out
        else:
            state_out = None

        vf_last_layer = maybe_td(tf.keras.layers.Conv2D(
                    filters=final_pi_filter_amt,
                    kernel_size=[3, 3],
                    strides=1,
                    activation=conv_activation,
                    padding="same",
                    name="{}_conv_{}".format('vf', "last")))(vf_last_layer)

        if parameter_noise:
            # assuming inputs shape (batch_size x w x h x channel)
            vf_last_layer = maybe_td(tf.keras.layers.LayerNormalization(
                axis=(1, 2),
                name=f"vf_LayerNorm_last"
            ))(vf_last_layer)

        print(f"action space n: {action_space.n}, rows: {rows}, columns: {colums}, filters: {int(action_space.n / (rows * colums))}")

        unmasked_logits_out = maybe_td(tf.keras.layers.Conv2D(
            filters=int(action_space.n / (rows * colums)),
            kernel_size=[3, 3],
            strides=1,
            activation=None,
            padding="same",
            name="{}_conv_{}".format('vf', "unmasked_logits")))(vf_last_layer)

        # vf_last_layer = maybe_td(tf.keras.layers.Conv2D(
        #     filters=1,
        #     kernel_size=[1, 1],
        #     strides=1,
        #     activation=conv_activation,
        #     padding="same",
        #     name="{}_conv_{}".format('vf', "last")))(vf_last_layer)
        #
        # vf_last_layer = maybe_td(tf.keras.layers.Flatten(name="vf_flatten"))(vf_last_layer)
        #
        # value_out = maybe_td(tf.keras.layers.Dense(
        #     units=1,
        #     name="vf_out",
        #     activation=None,
        #     kernel_initializer=normc_initializer(0.01)))(vf_last_layer)
        
        model_inputs = [self.vf_obs_inputs]
        model_outputs = [unmasked_logits_out]

        if self.use_lstm:
            model_inputs += [seq_lens_in, *state_in]
            model_outputs += state_out

        self.base_model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

        print(self.base_model.summary())

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        if "internal_state" in input_dict["obs"]:
            self.internal_states = input_dict["obs"]["internal_state"]

        vf_obs_inputs = input_dict['obs'][self.vf_obs_key]
        self.valid_actions_masks = input_dict["obs"]["valid_actions_mask"]

        if self.use_lstm:
            policy_out, *state_out = self.base_model([
                add_time_dimension(vf_obs_inputs, seq_lens),
                seq_lens,
                *state])
            policy_out = tf.reshape(policy_out, tf.shape(self.valid_actions_masks))
        else:
            policy_out = self.base_model([vf_obs_inputs])
            state_out = state

        if self.mask_invalid_actions:
            # set policy logits for invalid actions to zero
            inf_mask = tf.maximum(tf.log(self.valid_actions_masks), tf.float32.min)
            self.masked_policy_logits = policy_out + inf_mask
        else:
            self.masked_policy_logits = policy_out

        self.masked_policy_logits = tf.reshape(self.masked_policy_logits, [-1, self.num_outputs])

        return self.masked_policy_logits, state_out

    # def value_function(self):
    #     # if self._use_q_fn:
    #     #     raise NotImplementedError
    #     vf = tf.reshape(self._value_out, [-1])
    #     return vf

    def get_q_value_distributions(self, model_out):
        logits = tf.expand_dims(tf.ones_like(model_out), -1)
        dist = tf.expand_dims(tf.ones_like(model_out), -1)
        return model_out, logits, dist

    def get_q_values(self, model_out):
        return model_out

    def get_state_value(self, model_out):
        raise NotImplementedError

    @override(ModelV2)
    def get_initial_state(self):
        if self.use_lstm:
            def make_initial_state():
                return [np.zeros(self._lstm_state_shape, np.float32),
                        np.zeros(self._lstm_state_shape, np.float32)]

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


ModelCatalog.register_custom_model(SPATIAL_STRATEGO_Q_MODEL, SpatialStrategoQModel)

