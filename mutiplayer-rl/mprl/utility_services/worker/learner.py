import logging
import time
from typing import List

import grpc
from minio import Minio

from mprl.utility_services.cloud_storage import DEFAULT_LOCAL_SAVE_PATH
from mprl.utility_services.protobuf.population_server_pb2 import Confirmation, PolicyInfo
from mprl.utility_services.worker.base_interface import BaseClientManagerInterface, WorkerType, \
    _INFINITE_RETRY_INTERVAL_SECONDS, FalseConfirmationError

logger = logging.getLogger(__name__)


class LearnerManagerInterface(BaseClientManagerInterface):

    def __init__(self,
                 server_host: str,
                 port: int,
                 worker_id: str,
                 storage_client: Minio,
                 minio_bucket_name: str,
                 minio_local_dir: str = DEFAULT_LOCAL_SAVE_PATH
                 ):
        super(LearnerManagerInterface, self).__init__(
            server_host=server_host,
            port=port,
            worker_type=WorkerType.LEARNER,
            worker_id=worker_id,
            storage_client=storage_client,
            minio_bucket_name=minio_bucket_name,
            minio_local_dir=minio_local_dir)

    def submit_new_policy_for_population(self,
                                         policy_weights_key: str,
                                         policy_config_key: str,
                                         policy_class_name: str,
                                         policy_tags: List[str],
                                         infinite_retry_on_error: bool = True):
        while True:
            try:
                request = PolicyInfo()
                request.policy_key = policy_weights_key
                request.policy_model_config_key = policy_config_key
                request.policy_class_name = policy_class_name
                request.policy_tags.extend(policy_tags)
                response: Confirmation = self._stub.SubmitNewPolicyForPopulation(request)
                break
            except grpc.RpcError as err:
                if infinite_retry_on_error:
                    logger.warning(f"grpc.RPCError raised while getting submitting policy {policy_weights_key}:"
                                   f"\n{err}\n"
                                   f"(retrying in {_INFINITE_RETRY_INTERVAL_SECONDS} seconds)")
                    time.sleep(_INFINITE_RETRY_INTERVAL_SECONDS)
                else:
                    raise
        if not response.confirmation:
            raise FalseConfirmationError(f"Got '{request.confirmation}' from manager confirmation when "
                                                    f"submitting policy with key {policy_weights_key}"
                                                    f"message: {response.message}")
