import torch
import torch.nn as nn
from utils.counterfactual_utils import interchange_hook

class SequentialModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(SequentialModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.model(x)

class MLP_scaled(nn.Module):

    def __init__(self, input_size, output_size, scaled_depth):
        super(MLP_scaled, self).__init__()
        self.scaled_depth = int(scaled_depth)
        self.blocks = nn.ModuleList()

        self.input_size_list = [
            output_size,
            2*output_size,
            2*output_size,
            4*output_size,
            2*output_size,
            4*output_size,
            2*output_size,
            2*output_size,
            2*output_size,
            2*output_size,
            output_size,
            2*output_size,
            2*output_size
        ]

        # Create first block
        block = nn.ModuleList([SequentialModule(input_size, output_size) for _ in range(13)])
        self.blocks.append(block)

        # Create blocks based on scaled_depth
        for _ in range(1,self.scaled_depth):
            block = nn.ModuleList([SequentialModule(self.input_size_list[j], output_size) for j in range(13)])
            self.blocks.append(block)
        
        self.output = nn.Linear(output_size, 1)

        # self.loss = RMSELoss()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self,
                input_ids,
                # for interchange.
                interchanged_variables=None, 
                variable_names=None,
                interchanged_activations=None,
                # # losses
                t_outputs=None,
                causal_t_outputs=None,
                s_outputs=None
                ):
        student_output = {}
        student_output["hidden_states"] = [None] * 13 * self.scaled_depth

        x = torch.cat([input_ids[0:10], input_ids[12:]]) # Exclude CHO and insulin without scaling # Exclude CHO without scaling
        hooks = []
        # Process first block
        for module_index, module in enumerate(self.blocks[0]):
            hooks, student_output = interchange_intervention(
                x,
                interchanged_variables, 
                variable_names,
                interchanged_activations,
                module_index,
                module,
                hooks,
                student_output
            )

        if self.scaled_depth > 1:
            for j in range(1,self.scaled_depth):
                for i, module in enumerate(self.blocks[j]):
                    module_index += 1
                    if i in [0, 10]:
                        # Block N Layer 1 or Layer 11
                        hooks, student_output = interchange_intervention(
                            student_output["hidden_states"][module_index-13],
                            interchanged_variables, 
                            variable_names,
                            interchanged_activations,
                            module_index,
                            module,
                            hooks,
                            student_output
                        )
                    elif i in [1, 2, 4, 6, 8, 11]:
                        # Block N Layer 2 or Layer 3 or Layer 5 or Layer 7 or Layer 9 or Layer 12
                        hooks, student_output = interchange_intervention(
                            torch.cat((student_output["hidden_states"][module_index-14],
                                       student_output["hidden_states"][module_index-13]),
                                       dim=0),
                            interchanged_variables, 
                            variable_names,
                            interchanged_activations,
                            module_index,
                            module,
                            hooks,
                            student_output
                        )
                    elif i == 3:
                        # Block N Layer 4
                        hooks, student_output = interchange_intervention(
                            torch.cat((student_output["hidden_states"][module_index-14],
                                       student_output["hidden_states"][module_index-13],
                                       student_output["hidden_states"][module_index-12],
                                       student_output["hidden_states"][module_index-8]),
                                       dim=0),
                            interchanged_variables, 
                            variable_names,
                            interchanged_activations,
                            module_index,
                            module,
                            hooks,
                            student_output
                        )
                    elif i == 5:
                        # Block N Layer 6
                        hooks, student_output = interchange_intervention(
                            torch.cat((student_output["hidden_states"][module_index-13],
                                       student_output["hidden_states"][module_index-9],
                                       student_output["hidden_states"][module_index-8],
                                       student_output["hidden_states"][module_index-7]),
                                       dim=0),
                            interchanged_variables, 
                            variable_names,
                            interchanged_activations,
                            module_index,
                            module,
                            hooks,
                            student_output
                        )
                    elif i == 7:
                        # Block N Layer 8
                        hooks, student_output = interchange_intervention(
                            torch.cat((student_output["hidden_states"][module_index-15],
                                       student_output["hidden_states"][module_index-13]),
                                       dim=0),
                            interchanged_variables, 
                            variable_names,
                            interchanged_activations,
                            module_index,
                            module,
                            hooks,
                            student_output
                        )
                    elif i == 9:
                        # Block N Layer 10
                        hooks, student_output = interchange_intervention(
                            torch.cat((student_output["hidden_states"][module_index-17],
                                        student_output["hidden_states"][module_index-13]),
                                        dim=0),
                            interchanged_variables, 
                            variable_names,
                            interchanged_activations,
                            module_index,
                            module,
                            hooks,
                            student_output
                        )
                    elif i == 12:
                        # Block N Layer 10
                        hooks, student_output = interchange_intervention(
                            torch.cat((student_output["hidden_states"][module_index-22],
                                        student_output["hidden_states"][module_index-13]),
                                        dim=0),
                            interchanged_variables, 
                            variable_names,
                            interchanged_activations,
                            module_index,
                            module,
                            hooks,
                            student_output
                        )
                    else:
                        raise Exception("Index wrong for forward in MLP scaled")
        
        student_output["outputs"] = self.output(student_output["hidden_states"][-1])

        # IIT Objective
        # For each intermediate variable Yw ∈ {YTL, YTR, YBL, YBR}, we introduce an IIT
        #    objective that optimizes for N implementing Cw the
        #    submodel of C where the three intermediate variables
        #    that aren’t Yw are marginalized out:
        #     sum[ CE(Cw intinv, N intinv)]
        
        if causal_t_outputs is None:
            # if it is None, it is simply a forward for getting hidden states!
            if t_outputs is not None:
                s_outputs = student_output["outputs"]
                loss = self.loss(s_outputs, t_outputs.unsqueeze(0))
                student_output["loss"] = loss
        else:
            # causal loss.
            causal_s_outputs = student_output["outputs"]
            loss = self.loss(causal_s_outputs, causal_t_outputs.unsqueeze(0))
            student_output["loss"] = loss

            # measure the efficacy of the interchange.
            teacher_interchange_efficacy = (
                self.loss(
                    causal_t_outputs.unsqueeze(0),
                    t_outputs.unsqueeze(0),
                )
            )

            student_interchange_efficacy = (
                self.loss(
                    causal_s_outputs,
                    s_outputs,
                )
            )
            student_output["teacher_interchange_efficacy"] = teacher_interchange_efficacy
            student_output["student_interchange_efficacy"] = student_interchange_efficacy
        
        for h in hooks:
            h.remove()

        return student_output

def interchange_intervention(x,
                            interchanged_variables, 
                            variable_names,
                            interchanged_activations,
                            i,
                            layer_module,
                            hooks,
                            student_output):
    if variable_names != None and i in variable_names:
        assert interchanged_variables != None
        for interchanged_variable in variable_names[i]:
            interchanged_activations = interchanged_variables[interchanged_variable[0]]
            #https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#method-3-attach-a-hook AND interchange_with_activation_at()
            hook = layer_module.register_forward_hook(interchange_hook(interchanged_variable, interchanged_activations))
            hooks.append(hook)
    x = layer_module(
        x
    )
    student_output["hidden_states"][i] = x

    return hooks, student_output
