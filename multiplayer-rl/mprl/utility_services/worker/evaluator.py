import logging
import time

import grpc
from google.protobuf.empty_pb2 import Empty
from minio import Minio

from mprl.utility_services.cloud_storage import DEFAULT_LOCAL_SAVE_PATH
from mprl.utility_services.payoff_table import PolicySpec
from mprl.utility_services.protobuf.population_server_pb2 import Confirmation, EvalMatchupOrder, EvalMatchupResult
from mprl.utility_services.worker.base_interface import BaseClientManagerInterface, WorkerType, \
    _INFINITE_RETRY_INTERVAL_SECONDS, FalseConfirmationError

logger = logging.getLogger(__name__)


class EvaluatorManagerInterface(BaseClientManagerInterface):

    def __init__(self,
                 server_host: str,
                 port: int,
                 worker_id: str,
                 storage_client: Minio,
                 minio_bucket_name: str,
                 minio_local_dir: str = DEFAULT_LOCAL_SAVE_PATH
                 ):
        super(EvaluatorManagerInterface, self).__init__(
            server_host=server_host,
            port=port,
            worker_type=WorkerType.EVALUATOR,
            worker_id=worker_id,
            storage_client=storage_client,
            minio_bucket_name=minio_bucket_name,
            minio_local_dir=minio_local_dir)

    def get_eval_matchup(self, infinite_retry_on_error: bool = True):
        while True:
            try:
                request = Empty()
                response: EvalMatchupOrder = self._stub.RequestEvalMatchup(request)
                break
            except grpc.RpcError as err:
                if infinite_retry_on_error:
                    logger.warning(f"grpc.RPCError raised while getting eval matchup:\n{err}\n"
                                   f"(retrying in {_INFINITE_RETRY_INTERVAL_SECONDS} seconds)")
                    time.sleep(_INFINITE_RETRY_INTERVAL_SECONDS)
                else:
                    raise
        if response.no_matchups_needed:
            # logger.info("No matchups needed at the moment")
            return None

        as_policy = PolicySpec(policy_key=response.as_policy.policy_key,
                               policy_class_name=response.as_policy.policy_class_name,
                               policy_config_key=response.as_policy.policy_model_config_key,
                               tags=response.as_policy.policy_tags)

        against_policy = PolicySpec(policy_key=response.against_policy.policy_key,
                                    policy_class_name=response.against_policy.policy_class_name,
                                    policy_config_key=response.against_policy.policy_model_config_key,
                                    tags=response.against_policy.policy_tags)

        return {"as_policy": as_policy, "against_policy": against_policy, "num_games": response.num_games_to_play}

    def submit_eval_matchup_result(self,
                                   as_policy_key: str,
                                   against_policy_key: str,
                                   as_policy_avg_payoff: float,
                                   games_played: int,
                                   infinite_retry_on_error: bool = True):
        while True:
            try:
                request = EvalMatchupResult()
                request.as_policy_key = as_policy_key
                request.against_policy_key = against_policy_key
                request.payoff = as_policy_avg_payoff
                request.games_played = games_played
                response: Confirmation = self._stub.SubmitEvalMatchupResult(request)
                break
            except grpc.RpcError as err:
                if infinite_retry_on_error:
                    logger.warning(f"grpc.RPCError raised while getting submitting eval matchup result for\n"
                                   f"{as_policy_key}\nvs\n{against_policy_key}:"
                                   f"\n{err}\n"
                                   f"(retrying in {_INFINITE_RETRY_INTERVAL_SECONDS} seconds)")
                    time.sleep(_INFINITE_RETRY_INTERVAL_SECONDS)
                else:
                    raise
        if not response.confirmation:
            raise FalseConfirmationError(f"Got '{request.confirmation}' from manager confirmation when "
                                                    f"submitting eval matchup result for\n"
                                                    f"{as_policy_key}\nvs\n{against_policy_key}\n"
                                                    f"message: {response.message}")
