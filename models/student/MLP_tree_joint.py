import torch
import torch.nn as nn
from utils.counterfactual_utils import interchange_hook

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        input_size = 20
        num_neurons = 128
        output_size = 1
        # X_1
        self.X_1 = nn.Sequential(
            nn.Linear(input_size, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_11
        self.X_11 = nn.Sequential(
            nn.Linear(input_size, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_12
        self.X_12 = nn.Sequential(
            nn.Linear(input_size, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        # X_2
        self.X_2 = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_3
        self.X_3 = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        # X_6
        self.X_6_10 = nn.Sequential(
            nn.Linear(2*num_neurons, 2*num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_4
        self.X_4_5 = nn.Sequential(
            nn.Linear(2*num_neurons, 2*num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        # X_8
        self.X_8 = nn.Sequential(
            nn.Linear(2*num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_9
        self.X_9 = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        # X_13
        self.X_13 = nn.Sequential(
            nn.Linear(2*num_neurons, output_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        self.output = nn.Linear(output_size, output_size)

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
        student_output["hidden_states"]=[None] * 13
        # Interchange intervention
        x = torch.cat([input_ids[0:10], input_ids[12:]]) # Exclude CHO and insulin without scaling # Exclude CHO without scaling
        hooks = []
        
        # first level
        hooks, student_output, out1 = interchange_intervention(
            x,
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            0,
            self.X_1,
            hooks,
            student_output
        )
        hooks, student_output, out11 = interchange_intervention(
            x,
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            10,
            self.X_11,
            hooks,
            student_output
        )
        hooks, student_output, out12 = interchange_intervention(
            x,
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            11,
            self.X_12,
            hooks,
            student_output
        )


        # second level
        hooks, student_output, out2 = interchange_intervention(
            out1,
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            1,
            self.X_2,
            hooks,
            student_output
        )
        hooks, student_output, out6 = interchange_intervention(
            torch.cat((out11, out12)),
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            5,
            self.X_6_10,
            hooks,
            student_output
        )

        # third level
        hooks, student_output, out8 = interchange_intervention(
            out6,
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            7,
            self.X_8,
            hooks,
            student_output
        )

        # fourth level
        hooks, student_output, out3 = interchange_intervention(
            out2,
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            2,
            self.X_3,
            hooks,
            student_output
        )
        hooks, student_output, out9 = interchange_intervention(
            out8,
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            8,
            self.X_9,
            hooks,
            student_output
        )

        # fifth level
        hooks, student_output, out4 = interchange_intervention(
            torch.cat((out3, out9)),
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            3,
            self.X_4_5,
            hooks,
            student_output
        )
        
        
        # sixth level
        hooks, student_output, out13 = interchange_intervention(
            out4,
            interchanged_variables, 
            variable_names,
            interchanged_activations,
            12,
            self.X_13,
            hooks,
            student_output
        )

        
        student_output["outputs"] = self.output(out13)

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

    return hooks, student_output, x
