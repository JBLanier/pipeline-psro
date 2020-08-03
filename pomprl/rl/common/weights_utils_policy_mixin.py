from dill import load, dump

from pomprl.util import ensure_dir

class WeightsUtilsPolicyMixin(object):

    def get_model_weights(self, remove_scope_prefix:str = None):

        if remove_scope_prefix and not remove_scope_prefix.endswith("/"):
            remove_scope_prefix += "/"

        weights_dict = {}

        for var_name, var in self._variables.variables.items():

            if remove_scope_prefix:
                if var_name.startswith(remove_scope_prefix):
                    var_name = var_name[len(remove_scope_prefix):]
                else:
                    raise ValueError("Variable {} does not contain the scope prefix {}"
                                     .format(var_name, remove_scope_prefix))

            weights_dict[var_name] = var.eval(session=self._sess)

        return weights_dict

    def set_model_weights(self, weights, add_scope_prefix:str = None):

        if add_scope_prefix and not add_scope_prefix.endswith("/"):
            add_scope_prefix += "/"
        elif not add_scope_prefix:
            add_scope_prefix = ""


        current_weights = self.get_model_weights(remove_scope_prefix=add_scope_prefix)

        # for sync_key in weights.keys():
        #     assert sync_key in current_weights.keys(), f"sync key {sync_key} not in current weights"

        # for current_key in current_weights.keys():
        #     assert current_key in weights.keys(), f"current key {current_key} not in sync weights"
        # assert len(weights.keys()) > 0

        self._variables.set_weights(new_weights={
            add_scope_prefix + var_name: var for var_name, var in weights.items()
        })

    def save_model_weights(self, save_file_path:str, remove_scope_prefix:str = None):

        weights = self.get_model_weights(remove_scope_prefix=remove_scope_prefix)

        ensure_dir(save_file_path)

        with open(save_file_path, "wb") as dill_file:
            dump(obj=weights, file=dill_file)

    def load_model_weights(self, load_file_path:str, add_scope_prefix:str = None):
        with open(load_file_path, "rb") as dill_file:
            weights = load(file=dill_file)
        self.set_model_weights(weights=weights, add_scope_prefix=add_scope_prefix)

    def save_model_config_to_json(self, save_file_path: str):
        self.model.save_config_to_json(save_file_path=save_file_path)

