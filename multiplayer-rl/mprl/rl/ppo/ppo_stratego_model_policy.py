from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, KLCoeffMixin, ValueNetworkMixin, LearningRateSchedule, EntropyCoeffSchedule
from mprl.rl.common.weights_utils_policy_mixin import WeightsUtilsPolicyMixin




PPOStrategoModelTFPolicy = PPOTFPolicy.with_updates(
    name="PPOStrategoModelTFPolicy",
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, WeightsUtilsPolicyMixin
    ]
)