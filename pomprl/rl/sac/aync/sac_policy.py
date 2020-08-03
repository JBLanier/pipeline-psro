from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Box, Discrete
import numpy as np
import logging
from pomprl.rl.common.weights_utils_policy_mixin import WeightsUtilsPolicyMixin

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.sac.sac_model import SACModel
from ray.rllib.agents.ddpg.noop_model import NoopModel
from ray.rllib.agents.dqn.dqn_policy import _postprocess_dqn, PRIO_WEIGHTS
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils import try_import_tf, try_import_tfp
from ray.rllib.utils.tf_ops import minimize_and_clip, make_tf_callable
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.models.model import restore_original_dimensions
from pomprl.rl.sac.sac import DEFAULT_CONFIG

tf = try_import_tf()
tfp = try_import_tfp()
logger = logging.getLogger(__name__)


def build_sac_model(policy, obs_space, action_space, config):

    if not isinstance(action_space, Discrete):
        raise UnsupportedSpaceException

    num_outputs = action_space.n

    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        num_outputs,
        config["model"],
        framework="tf",
        name="sac_model")

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        num_outputs,
        config["model"],
        framework="tf",
        name="target_sac_model")

    return policy.model


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    return _postprocess_dqn(policy, sample_batch)


def build_action_output(policy, model, input_dict, obs_space, action_space,
                        config):

    print(f"\n\n\ninput dict: {input_dict}\n\n\n")

    base_model_out, _ = model({
        "obs": input_dict[SampleBatch.CUR_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, policy._state_in, policy._seq_lens)

    stochastic_actions, log_pis = model.get_policy_output(
        base_model_out, deterministic=False)
    deterministic_actions, _ = model.get_policy_output(
        base_model_out, deterministic=True)

    # stochastic_actions = tf.Print(stochastic_actions, [policy._state_in], message="state in: ")
    # stochastic_actions = tf.Print(stochastic_actions, [policy._seq_lens], message="seq_lens: ")
    # stochastic_actions = tf.Print(stochastic_actions, [input_dict[SampleBatch.CUR_OBS]], message="obs: ")

    actions = tf.cond(policy.stochastic, lambda: stochastic_actions,
                      lambda: deterministic_actions)
    action_probabilities = tf.cond(policy.stochastic, lambda: log_pis,
                                   lambda: tf.zeros_like(log_pis))
    policy.output_actions = actions
    return actions, action_probabilities


def actor_critic_loss(policy, model, _, train_batch):

    restored_obs = restore_original_dimensions(
            train_batch[SampleBatch.CUR_OBS], model.obs_space, "tf")

    flat_valid_actions_mask_t = tf.reshape(restored_obs["valid_actions_mask"], [-1, policy.model.num_outputs])

    restored_next_obs = restore_original_dimensions(
        train_batch[SampleBatch.NEXT_OBS], model.obs_space, "tf")

    zero_states = []
    i = 0
    while "state_in_{}".format(i) in train_batch:
        zero_states.append(tf.zeros_like(train_batch["state_in_{}".format(i)]))
        i += 1

    model_out_t, state_out_t = model({
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, zero_states, train_batch['seq_lens'])

    if state_out_t:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        valid_mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        valid_mask = tf.reshape(valid_mask, [-1])
    else:
        valid_mask = tf.ones_like(train_batch[SampleBatch.CUR_OBS], dtype=tf.bool)

    def reduce_mean_valid(t):
        return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

    model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, zero_states, train_batch['seq_lens'])

    target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, zero_states, train_batch['seq_lens'])

    policy_probs_t = tf.nn.softmax(model.get_policy_logits(base_model_out=model_out_t), axis=-1)  # shape (batchsize, num_actions)
    log_pis_t = tf.log(policy_probs_t + 1e-8)  # shape (batchsize, num_actions)

    policy_probs_tp1 = tf.nn.softmax(model.get_policy_logits(base_model_out=model_out_tp1), axis=-1)  # shape (batchsize, num_actions)
    log_pis_tp1 = tf.log(policy_probs_tp1 + 1e-8)  # shape (batchsize, num_actions)

    log_alpha = model.log_alpha
    alpha = model.alpha

    # q network evaluation
    main_q_t = model.get_q_values(base_model_out=model_out_t)  # shape (batchsize, num_actions)
    action_indices = tf.stack(
        [tf.cast(tf.range(tf.shape(main_q_t)[0]), tf.int64), train_batch[SampleBatch.ACTIONS]], axis=-1)
    main_q_t_selected = tf.gather_nd(main_q_t, action_indices)  # shape (batchsize,)

    if policy.config["twin_q"]:
        twin_q_t = model.get_twin_q_values(base_model_out=model_out_t)  # shape (batchsize, num_actions)
        twin_q_t_selected = tf.gather_nd(twin_q_t, action_indices)  # shape (batchsize,)
        min_q_t = tf.math.minimum(main_q_t, twin_q_t)  # shape (batchsize, num_actions)
    else:
        min_q_t = main_q_t  # shape (batchsize, num_actions)

    # target q network evaluation
    main_q_targetnet_tp1 = policy.target_model.get_q_values(base_model_out=target_model_out_tp1)  # shape (batchsize, num_actions)
    if policy.config["twin_q"]:
        twin_q_targetnet_tp1 = policy.target_model.get_twin_q_values(base_model_out=target_model_out_tp1)  # shape (batchsize, num_actions)
        min_q_targetnet_tp1 = tf.math.minimum(main_q_targetnet_tp1, twin_q_targetnet_tp1)  # shape (batchsize, num_actions)
    else:
        min_q_targetnet_tp1 = main_q_targetnet_tp1  # shape (batchsize, num_actions)

    # todo just changed this line from log_pis_t to log_pis_tp1
    q_value_tp1_per_action = (min_q_targetnet_tp1 - alpha * log_pis_tp1)
    value_tp1 = tf.stop_gradient(tf.reduce_sum(policy_probs_tp1 * q_value_tp1_per_action, axis=-1))  # shape (batchsize,)
    assert np.array_equal(np.asarray(value_tp1.get_shape().as_list()), [None]), f"shape is {np.asarray(value_tp1.get_shape().as_list())}"

    value_tp1_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * value_tp1   # shape (batchsize,)

    assert policy.config["n_step"] == 1, "TODO(hartikainen) n_step > 1"

    # compute RHS of bellman equation
    q_t_target = train_batch[SampleBatch.REWARDS] + policy.config["gamma"] * value_tp1_masked   # shape (batchsize,)
    assert np.array_equal(np.asarray(q_t_target.get_shape().as_list()), [None])

    q1_loss = 0.5 * reduce_mean_valid((main_q_t_selected - q_t_target)**2)
    q2_loss = 0.5 * reduce_mean_valid((twin_q_t_selected - q_t_target) ** 2)

    # TODO use a baseline? hw from levine, page 6 http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5b.pdf
    baseline=0.0
    value_t = tf.stop_gradient(tf.reduce_sum(policy_probs_t * (min_q_t - alpha * log_pis_t), axis=-1, keep_dims=True))  # shape (batchsize, 1)
    # baseline = value_t

    actor_loss_per_batch_element = tf.reduce_sum(policy_probs_t * tf.stop_gradient(alpha * log_pis_t - min_q_t + baseline), axis= -1)  # shape (batchsize,)
    actor_loss = reduce_mean_valid(actor_loss_per_batch_element)

    target_entropies = policy.config["max_entropy_target_proportion"] * tf.log(1e-8 + tf.reduce_sum(flat_valid_actions_mask_t, axis=-1, keep_dims=False)) if policy.config["target_entropy"] == "auto" else policy.config["target_entropy"]  # shape (batchsize,) if 'auto'
    assert np.array_equal(np.asarray(target_entropies.get_shape().as_list()), [None])

    target_entropies = tf.debugging.check_numerics(
        target_entropies, f"nan found in target_entropies", name=None)

    policy.target_entropies = target_entropies

    # assert False, f"target entropy is {target_entropy}, action_space.n is {policy.action_space.n}"

    pi_entropies = -tf.reduce_sum(policy_probs_t * log_pis_t, axis=-1)

    pi_entropies = tf.debugging.check_numerics(
        pi_entropies, f"nan found in pi_entropies", name=None)

    # assert np.array_equal(np.asarray(pi_entropies.get_shape().as_list()), [None])
    policy.pi_entropies = pi_entropies
    alpha_backup = tf.stop_gradient(target_entropies - pi_entropies)  # shape (batchsize,)
    assert np.array_equal(np.asarray(alpha_backup.get_shape().as_list()), [None]), f"actual shape {alpha_backup.get_shape().as_list()}"
    alpha_loss = -reduce_mean_valid(log_alpha * alpha_backup)


    # alpha_loss = tf.Print(alpha_loss, [alpha_loss], "\n\n\n\n\n\n\n\n\n\n\n\n\n\nalpha loss is:")
    # alpha_loss = tf.debugging.check_numerics(
    #     alpha_loss, f"nan found in alpha loss", name=None)


    # alpha_loss = tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(policy_t) * (-alpha * tf.stop_gradient(log_pis_t + target_entropies)), axis=-1))

    # save for stats function
    policy.min_q_t = min_q_t
    # policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.q1_loss = q1_loss
    policy.q2_loss = q2_loss

    policy.alpha_loss = alpha_loss

    total_loss = actor_loss + q1_loss + q2_loss + alpha_loss

    total_loss = tf.debugging.check_numerics(
        total_loss, f"nan found in total_loss", name=None)

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    return total_loss


def gradients(policy, optimizer, loss):
    if policy.config["grad_norm_clipping"] is not None:
        raise NotImplementedError
        actor_grads_and_vars = minimize_and_clip(
            optimizer,
            policy.actor_loss,
            var_list=policy.model.policy_variables(),
            clip_val=policy.config["grad_norm_clipping"])
        # if policy.config["twin_q"]:
        #     critic_grads_and_vars = []
        #     critic_grads_and_vars += minimize_and_clip(
        #         optimizer,
        #         policy.critic_loss[0],
        #         var_list=policy.model.main_q_variables(),
        #         clip_val=policy.config["grad_norm_clipping"])
        #     critic_grads_and_vars += minimize_and_clip(
        #         optimizer,
        #         policy.critic_loss[1],
        #         var_list=policy.model.twin_q_variables(),
        #         clip_val=policy.config["grad_norm_clipping"])
        # else:
        #     critic_grads_and_vars = minimize_and_clip(
        #         optimizer,
        #         policy.critic_loss[0],
        #         var_list=policy.model.q_variables(),
        #         clip_val=policy.config["grad_norm_clipping"])
        alpha_grads_and_vars = minimize_and_clip(
            optimizer,
            policy.alpha_loss,
            var_list=[policy.model.log_alpha],
            clip_val=policy.config["grad_norm_clipping"])
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.model.policy_variables())

        critic_grads_and_vars = policy._critic_optimizer.compute_gradients(
            policy.q1_loss, var_list=policy.model.main_q_variables())

        critic_grads_and_vars2 = policy._critic_optimizer2.compute_gradients(
            policy.q2_loss, var_list=policy.model.twin_q_variables())

        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
            policy.alpha_loss, var_list=[policy.model.log_alpha])

    # save these for later use in build_apply_op
    policy._actor_grads_and_vars = [(g, v) for (g, v) in actor_grads_and_vars
                                    if g is not None]
    policy._critic_grads_and_vars = [(g, v) for (g, v) in critic_grads_and_vars
                                     if g is not None]

    policy._critic_grads_and_vars2 = [(g, v) for (g, v) in critic_grads_and_vars2
                                     if g is not None]

    # print_op = tf.Print(alpha_grads_and_vars[0][0], [alpha_grads_and_vars[0][0]], "\n\n\n\n\n\n\n\n\n\n\n\n\n\nalpha grad is:")

    # with tf.control_dependencies([print_op]):
    policy._alpha_grads_and_vars = [(g, v) for (g, v) in alpha_grads_and_vars
                                    if g is not None]

    grads_and_vars = (
        policy._actor_grads_and_vars +
        policy._critic_grads_and_vars + policy._critic_grads_and_vars +
        policy._alpha_grads_and_vars
    )
    # assert False, f"alpha grads and vars:\n{policy._alpha_grads_and_vars}"

    return grads_and_vars

def apply_gradients(policy, optimizer, grads_and_vars):
    actor_apply_ops = policy._actor_optimizer.apply_gradients(
        policy._actor_grads_and_vars)

    critic_apply_ops = policy._critic_optimizer.apply_gradients(
        policy._critic_grads_and_vars)

    critic_apply_ops2 = policy._critic_optimizer2.apply_gradients(
        policy._critic_grads_and_vars2)

    alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
        policy._alpha_grads_and_vars,
        global_step=tf.train.get_or_create_global_step())

    return tf.group([
                    actor_apply_ops,
                     alpha_apply_ops,
                     critic_apply_ops, critic_apply_ops2
                     ])


# def gradients(policy, optimizer, loss):
#     if policy.config["grad_norm_clipping"] is not None:
#         actor_grads_and_vars = minimize_and_clip(
#             optimizer,
#             policy.actor_loss,
#             var_list=policy.model.policy_variables(),
#             clip_val=policy.config["grad_norm_clipping"])
#         critic_grads_and_vars = minimize_and_clip(
#             optimizer,
#             policy.critic_loss,
#             var_list=policy.model.q_variables(),
#             clip_val=policy.config["grad_norm_clipping"])
#         alpha_grads_and_vars = minimize_and_clip(
#             optimizer,
#             policy.alpha_loss,
#             var_list=[policy.model.log_alpha],
#             clip_val=policy.config["grad_norm_clipping"])
#     else:
#         actor_grads_and_vars = optimizer.compute_gradients(
#             policy.actor_loss, var_list=policy.model.policy_variables())
#         critic_grads_and_vars = optimizer.compute_gradients(
#             policy.critic_loss, var_list=policy.model.q_variables())
#         alpha_grads_and_vars = optimizer.compute_gradients(
#             policy.alpha_loss, var_list=[policy.model.log_alpha])
#     # save these for later use in build_apply_op
#     policy._actor_grads_and_vars = [(g, v) for (g, v) in actor_grads_and_vars
#                                     if g is not None]
#     policy._critic_grads_and_vars = [(g, v) for (g, v) in critic_grads_and_vars
#                                      if g is not None]
#     policy._alpha_grads_and_vars = [(g, v) for (g, v) in alpha_grads_and_vars
#                                     if g is not None]
#     grads_and_vars = (
#         policy._actor_grads_and_vars + policy._critic_grads_and_vars +
#         policy._alpha_grads_and_vars)
#     return grads_and_vars


def stats(policy, train_batch):
    stats = {
        # "td_error": tf.reduce_mean(policy.td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "q1_loss": tf.reduce_mean(policy.q1_loss),
        "q2_loss": tf.reduce_mean(policy.q2_loss),
        "mean_q": tf.reduce_mean(policy.min_q_t),
        "max_q": tf.reduce_max(policy.min_q_t),
        "min_q": tf.reduce_min(policy.min_q_t),
        "alpha_loss": policy.alpha_loss,
        "alpha": policy.model.alpha,
        "mean_target_entropy": tf.reduce_mean(policy.target_entropies, axis=0),
        "mean_actual_entropy": tf.reduce_mean(policy.pi_entropies, axis=0)
    }
    return stats

class ExplorationStateMixin(object):
    def __init__(self, obs_space, action_space, config):
        self.stochastic = tf.get_variable(
            initializer=tf.constant_initializer(config["exploration_enabled"]),
            name="stochastic",
            shape=(),
            trainable=False,
            dtype=tf.bool)

    def set_epsilon(self, epsilon):
        pass


class ActorCriticOptimizerMixin(object):
    def __init__(self, config):
        # create global step for counting the number of update operations
        self.global_step = tf.train.get_or_create_global_step()

        # use separate optimizers for actor & critic
        self._actor_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["actor_learning_rate"])
        self._critic_optimizer = tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["critic_learning_rate"])
        self._critic_optimizer2 = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["critic_learning_rate"])
        self._alpha_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["entropy_learning_rate"])


class ComputeTDErrorMixin(object):
    def __init__(self):
        @make_tf_callable(self.get_session(), dynamic_shape=True)
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            if not self.loss_initialized():
                return tf.zeros_like(rew_t)

            # Do forward pass on loss to update td error attribute
            actor_critic_loss(
                self, self.model, None, {
                    SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_t),
                    SampleBatch.ACTIONS: tf.convert_to_tensor(act_t),
                    SampleBatch.REWARDS: tf.convert_to_tensor(rew_t),
                    SampleBatch.NEXT_OBS: tf.convert_to_tensor(obs_tp1),
                    SampleBatch.DONES: tf.convert_to_tensor(done_mask),
                    PRIO_WEIGHTS: tf.convert_to_tensor(importance_weights),
                })

            return self.td_error

        self.compute_td_error = compute_td_error


class TargetNetworkMixin(object):
    def __init__(self, config):
        @make_tf_callable(self.get_session())
        def update_target_fn(tau):
            tau = tf.convert_to_tensor(tau, dtype=tf.float32)
            update_target_expr = []
            model_vars = self.model.trainable_variables()
            target_model_vars = self.target_model.trainable_variables()
            assert len(model_vars) == len(target_model_vars), \
                (model_vars, target_model_vars)
            for var, var_target in zip(model_vars, target_model_vars):
                update_target_expr.append(
                    var_target.assign(tau * var + (1.0 - tau) * var_target))
                logger.debug("Update target op {}".format(var_target))
            return tf.group(*update_target_expr)

        # Hard initial update
        self._do_update = update_target_fn
        self.update_target(tau=1.0)

    # support both hard and soft sync
    def update_target(self, tau=None):
        self._do_update(np.float32(tau or self.config.get("tau")))

    # @override(TFPolicy)
    # def variables(self):
    #     return self.model.variables() + self.target_model.variables()


def setup_early_mixins(policy, obs_space, action_space, config):
    ExplorationStateMixin.__init__(policy, obs_space, action_space, config)
    ActorCriticOptimizerMixin.__init__(policy, config)


def setup_mid_mixins(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)


SACDiscreteTFPolicy = build_tf_policy(
    name="SACDiscreteTFPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    make_model=build_sac_model,
    postprocess_fn=postprocess_trajectory,
    action_sampler_fn=build_action_output,
    loss_fn=actor_critic_loss,
    stats_fn=stats,
    gradients_fn=gradients,
    apply_gradients_fn=apply_gradients,
    # extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error},
    mixins=[
        TargetNetworkMixin, ExplorationStateMixin, ActorCriticOptimizerMixin,
        ComputeTDErrorMixin, WeightsUtilsPolicyMixin
    ],
    before_init=setup_early_mixins,
    before_loss_init=setup_mid_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False)
