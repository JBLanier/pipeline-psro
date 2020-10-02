
import logging

logger = logging.getLogger(__name__)

def create_reference_policy_update_callback_for_self_play_eval(trainable_policy_key: str, reference_policy_key: str,
                                                               new_policy_accept_score=None,
                                                               accept_metric=None,
                                                               accept_mode='gte',
                                                               accept_callback=None,
                                                               reset_trainable_policy_weights_on_failure=False):
    # Creates a callback that updates a reference policy if a trainable policy can beat it by a certain win percentage.
    # This allows for the same policy validation rules as in Alpha Zero

    def do_reference_policy_update_after_self_play_eval(trainer, eval_metrics):
        if new_policy_accept_score is None:
            accept_score = trainer.config['new_policy_accept_score']
        else:
            accept_score = new_policy_accept_score

        if isinstance(accept_score, str):
            accept_score = (accept_score,)

        if isinstance(accept_score, tuple):
            accept_score_keys = accept_score
            accept_score = eval_metrics
            for key in accept_score_keys:
                accept_score = accept_score[key]

        if accept_metric is None:
            new_policy_accept_metric = (trainable_policy_key, 'win_percentage')
        else:
            new_policy_accept_metric = accept_metric
            if not isinstance(new_policy_accept_metric, tuple):
                new_policy_accept_metric = (new_policy_accept_metric,)

        if not hasattr(trainer, "policy_version"):
            trainer.policy_version = 0

        new_policy_score = eval_metrics
        for key in new_policy_accept_metric:
            new_policy_score = new_policy_score[key]

        print("new policy score is", new_policy_score)

        accept_new_policy = False
        if accept_mode == 'gte':
            accept_new_policy = new_policy_score >= accept_score
        if accept_mode == 'lte':
            accept_new_policy = new_policy_score <= accept_score
        else:
            assert accept_mode == 'gte' or accept_mode == 'lte'

        if accept_new_policy:
            logger.info("ACCEPTING new reference policy")

            new_reference_weights = trainer.get_weights([trainable_policy_key])
            new_reference_weights[reference_policy_key] = new_reference_weights.pop(trainable_policy_key)
            assert trainable_policy_key not in new_reference_weights.keys()
            trainer.set_weights(new_reference_weights)

            trainer.policy_version += 1
            logger.info("Reference policy version is now v{}".format(trainer.policy_version))

            if accept_callback is not None:
                accept_callback(trainer)

        elif reset_trainable_policy_weights_on_failure:
            new_trainable_weights = trainer.get_weights([reference_policy_key])
            new_trainable_weights[trainable_policy_key] = new_trainable_weights.pop(reference_policy_key)
            assert reference_policy_key not in new_trainable_weights.keys()
            trainer.set_weights(new_trainable_weights)

        eval_metrics['current_reference_policy_version'] = trainer.policy_version

    return do_reference_policy_update_after_self_play_eval


# def create_swap_policies_to_train_callback(policy_key_a: str, policy_key_b: str):
#     # swaps one policy out with the other.
#     # a will be swapped out with b if present.
#     # b will be swapped out with a if present.
#
#     def swap_policies_to_train(trainer, eval_metrics):
#
#         def worker_swap_policies_to_train(worker, worker_index):
#             swap_completed = False
#             for existing_policy, new_policy in [(policy_key_a, policy_key_b), (policy_key_b, policy_key_a)]:
#                 if existing_policy in worker.policies_to_train:
#                     worker.policies_to_train[worker.policies_to_train.index(existing_policy)] = new_policy
#                     swap_completed = True
#
#                     if worker_index == 0:
#                         print("Swapped policy {} in to be trained instead of policy {}"
#                                      .format(new_policy, existing_policy))
#
#                     break
#             if not swap_completed:
#                 raise ValueError(
#                     "Neither policies ({} nor {}) were already in policies_to_train, so a swap couldn't be done."
#                     .format(policy_key_a, policy_key_b))
#
#         trainer.workers.foreach_worker_with_index(worker_swap_policies_to_train)
#
#     return swap_policies_to_train

# def create_swap_policies_to_train_callback(trainable_tmp_policy_key: str, static_tmp_policy_key: str):
#
#     def swap_policies_to_train(trainer):
#
#         if not hasattr(trainer, 'next_policy_to_swap_in'):
#             trainer.next_policy_to_swap_in = policy_key_b
#         next_policy_to_swap_in = trainer.next_policy_to_swap_in
#
#         if trainer.next_policy_to_swap_in == policy_key_b:
#             next_policy_to_swap_out = policy_key_a
#         else:
#             next_policy_to_swap_out = policy_key_b
#
#         newly_trained_policy_weights = trainer.get_weights([trainable_tmp_policy_key, static_tmp_policy_key])
#
#         copy_policies_in_trainer(trainer=trainer, src_key=trainable_tmp_policy_key, dst_key=next_policy_to_swap_out)
#         copy_policies_in_trainer(trainer=trainer, src_key=static_tmp_policy_key, dst_key=next_policy_to_swap_in)
#
#         print("swapping policies {} and {}".format(trainable_tmp_policy_key, static_tmp_policy_key))
#
#         # trainable policy is was what used statically in previous iteration
#         newly_trained_policy_weights[trainable_tmp_policy_key] = newly_trained_policy_weights[next_policy_to_swap_in]
#
#         # static policy is now the weights we just trained
#         newly_trained_policy_weights[static_tmp_policy_key] = newly_trained_policy_weights[next_policy_to_swap_out]
#
#         trainer.next_policy_to_swap_in = next_policy_to_swap_out
#
#         trainer.set_weights(newly_trained_policy_weights)
#
#     return swap_policies_to_train