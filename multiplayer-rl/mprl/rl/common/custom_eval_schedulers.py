from mprl.rl.common.util import set_policies_to_train_in_trainer


# def get_ed_fixed_scheduler(best_response_iterations, equilibrium_policy_iterations,
#                            trainable_tmp_policy_key, static_tmp_policy_key):
#
#     def ed_fixed_scheduler(trainer, eval_data):
#         if 'is_training_best_response' not in eval_data:
#             eval_data['is_training_best_response'] = True
#
#         if 'swap_policies_at_iteration' not in eval_data:
#             eval_data['swap_policies_at_iteration'] = trainer.iteration + best_response_iterations
#
#         if trainer.iteration == eval_data['swap_policies_at_iteration']:
#             swap_policies_in_trainer(trainer=trainer,
#                                      key_a=trainable_tmp_policy_key,
#                                      key_b=static_tmp_policy_key)
#             print("swapped policies {} and {}".format(trainable_tmp_policy_key, static_tmp_policy_key))
#
#             if eval_data['is_training_best_response']:
#                 # now train equilibrium policy
#                 swap_at_iteration = trainer.iteration + equilibrium_policy_iterations
#                 eval_data['swap_policies_at_iteration'] = trainer.iteration + equilibrium_policy_iterations
#                 print("Should swap to BR after iteration {}".format(swap_at_iteration))
#
#                 eval_data['is_training_best_response'] = False
#                 return True, None
#             else:
#                 # now train best response policy
#                 swap_at_iteration = trainer.iteration + best_response_iterations
#                 eval_data['swap_policies_at_iteration'] = trainer.iteration + best_response_iterations
#                 print("Should swap to equilibrium policy after iteration {}".format(swap_at_iteration))
#
#                 eval_data['is_training_best_response'] = True
#                 return False, None
#
#         return False, None
#
#     return ed_fixed_scheduler


def get_ed_adaptive_scheduler(best_response_threshold_score, threshold_mode, threshhold_metric,
                              equilibrium_policy_iterations,
                           best_response_policy_key, equilibrium_policy_key):

    def swap_policies_if_threshold_met_callback(trainer, eval_metrics, eval_data):
        br_threshold_score = best_response_threshold_score

        if isinstance(br_threshold_score, str):
            br_threshold_score = (br_threshold_score,)

        if isinstance(br_threshold_score, tuple):
            br_score_keys = br_threshold_score
            br_threshold_score = eval_metrics
            for key in br_score_keys:
                br_threshold_score = br_threshold_score[key]

        if callable(threshhold_metric):
            recorded_score = threshhold_metric(trainer)
        else:
            br_threshold_metric = threshhold_metric

            if not isinstance(br_threshold_metric, tuple):
                br_threshold_metric = (br_threshold_metric,)

            recorded_score = eval_metrics
            for key in br_threshold_metric:
                recorded_score = recorded_score[key]

        print("adaptive_br_score is", recorded_score)

        if not hasattr(trainer, "policy_version"):
            trainer.policy_version = 0

        br_accepted = False
        if threshold_mode == 'gte':
            br_accepted = recorded_score >= br_threshold_score
        if threshold_mode == 'lte':
            br_accepted = recorded_score <= br_threshold_score
        else:
            assert threshold_mode == 'gte' or threshold_mode == 'lte'

        if br_accepted:
            print("Best Response Accepted")
            # swap_policies_in_trainer(trainer=trainer,
            #                          key_a=trainable_tmp_policy_key,
            #                          key_b=static_tmp_policy_key)
            # print("swapped policies {} and {}".format(trainable_tmp_policy_key, static_tmp_policy_key))

            set_policies_to_train_in_trainer(trainer, [equilibrium_policy_key])
            print("set {} to be trained".format(equilibrium_policy_key))

            swap_back_at_iteration = trainer.iteration + equilibrium_policy_iterations
            print("Should swap back after iteration {}".format(swap_back_at_iteration))
            eval_data['swap_policies_at_iteration'] = swap_back_at_iteration
            eval_data['is_training_best_response'] = False

    def ed_adaptive_scheduler(trainer, eval_data):
        if 'is_training_best_response' not in eval_data:
            eval_data['is_training_best_response'] = True

        if 'swap_policies_at_iteration' not in eval_data:
            eval_data['swap_policies_at_iteration'] = None

        if eval_data['swap_policies_at_iteration'] is not None and \
                trainer.iteration == eval_data['swap_policies_at_iteration']:

            if eval_data['is_training_best_response']:
                assert False

            # now train best response policy
            # swap_policies_in_trainer(trainer=trainer,
            #                          key_a=trainable_tmp_policy_key,
            #                          key_b=static_tmp_policy_key)

            set_policies_to_train_in_trainer(trainer, [best_response_policy_key])
            print("set {} to be trained".format(best_response_policy_key))
            # print("swapped policies {} and {}".format(trainable_tmp_policy_key, static_tmp_policy_key))

            eval_data['swap_policies_at_iteration'] = None
            eval_data['is_training_best_response'] = True
            return False, None

        return True, lambda _trainer, eval_metrics: swap_policies_if_threshold_met_callback(trainer, eval_metrics, eval_data)

    return ed_adaptive_scheduler
