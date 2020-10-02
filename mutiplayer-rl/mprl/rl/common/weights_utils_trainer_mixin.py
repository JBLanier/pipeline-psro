import os

from ray.rllib.agents.trainer import Trainer
from ray.tune.trial import ExportFormat
from ray.tune import TuneError


class WeightsUtilsTrainerMixin(object):

    def _export_model(self, export_formats, export_dir):

        MODEL_WEIGHTS = "model_weights"

        try:
            ExportFormat.validate(export_formats)
        except TuneError as err:
            if MODEL_WEIGHTS in export_formats:
                idx = export_formats.index(MODEL_WEIGHTS)
                ExportFormat.validate(export_formats[:idx] + export_formats[idx+1:])
            else:
                raise err

        exported = {}
        if ExportFormat.CHECKPOINT in export_formats:
            path = os.path.join(export_dir, ExportFormat.CHECKPOINT)
            self.export_policy_checkpoint(path)
            exported[ExportFormat.CHECKPOINT] = path
        if ExportFormat.MODEL in export_formats:
            path = os.path.join(export_dir, ExportFormat.MODEL)
            self.export_policy_model(path)
            exported[ExportFormat.MODEL] = path

        if MODEL_WEIGHTS in export_formats:
            path = os.path.join(export_dir, MODEL_WEIGHTS)

            for policy_id in self.config["export_policy_weights_ids"]:
                policy_to_save = self.workers.local_worker().policy_map[policy_id]
                local_file_path = os.path.join(export_dir, MODEL_WEIGHTS, policy_id + ".dill")
                policy_to_save.save_model_weights(local_file_path, remove_scope_prefix=policy_id)
                exported[(MODEL_WEIGHTS, policy_id)] = path

        return exported

    def save_policy_model_configs_to_json(self):

        for policy_id, policy in self.workers.local_worker().policy_map.items():
            policy.save_model_config_to_json(save_file_path=os.path.join(self._logdir, policy_id+"_config.json"))

    def save_policy_weights_to_dir(self, dir):
        for policy_id, policy in self.workers.local_worker().policy_map.items():
            policy.save_model_weights(
                save_file_path=os.path.join(dir, f"{policy_id}_checkpoint.policy"),
                remove_scope_prefix=policy_id)
