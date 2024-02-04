import logging
import numpy as np
import torch


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def deserialize_variable_name(variable_name):
    deserialized_variables = []
    variable_list = variable_name.split("$")
    if "[" in variable_list[1]:
        left_l = int(variable_list[1].split(":")[1].strip("["))
        right_l = int(variable_list[1].split(":")[2].strip("]"))
    else:
        left_l = int(variable_list[1].split(":")[-1])
        right_l = int(variable_list[1].split(":")[-1])+1

    left_d = variable_list[2].split(",")[0].strip("[")
    right_d = variable_list[2].split(",")[1].strip("]")
    
    for i in range(left_l, right_l):
        deserialized_variable = f"$L:{i}$[{left_d},{right_d}]"
        deserialized_variables += [deserialized_variable]
    return deserialized_variables


def parse_variable_name(variable_name, model_config=None):
    """
    LOC locality of convolution
    """
    if model_config == None:
        variable_list = variable_name.split("$")
        layer_number = int(variable_list[1].split(":")[-1])
        LOC_left = variable_list[2].split(",")[0].strip("[")
        LOC_right = variable_list[2].split(",")[1].strip("]")
        LOC = np.s_[LOC_left:LOC_right]
        return layer_number, LOC
    else:
        # to be supported.
        pass
    
def get_activation_at(
    model, input_ids,
    variable_names,
    look_up = None,
    labels=None
):
    if look_up is not None:
        # Teacher
        outputs = model(
            input_ids, look_up=look_up
        )
    else:
        outputs = model(
            input_ids
        )
    activations = []
    for variable in variable_names:
        layer_index, LOC = parse_variable_name(
            variable_name=variable
        )
        # TODO NOT REUSABLE AT ALL, NEED TO GENERALIZE
        if LOC.start == LOC.stop and LOC.start==":":
            head_slice = outputs["hidden_states"][layer_index]
        else:
            assert LOC.stop==':'
            start_index = int(LOC.start)
            head_slice = outputs["hidden_states"][layer_index][start_index,:]
            
        activations += [head_slice]
    return activations

def interchange_hook(interchanged_variable, interchanged_activations):
    # the hook signature
    def hook(model, input, output):
        # interchange inplace.
        # TODO NOT REUSABLE AT ALL, NEED TO GENERALIZE
        assert interchanged_variable[1].stop==':'
        if interchanged_variable[1].start == ':':
            output = interchanged_activations
        else:
            output[int(interchanged_variable[1].start),:] = interchanged_activations
        return output
    return hook