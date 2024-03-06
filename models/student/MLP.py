import torch
import torch.nn as nn
from utils.counterfactual_utils import interchange_hook

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        num_neurons = 1
        output_size = 1
        # X_1
        self.X_1 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_2
        self.X_2 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_3
        self.X_3 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_4
        self.X_4 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_5
        self.X_5 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_6
        self.X_6 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_7
        self.X_7 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_8
        self.X_8 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_9
        self.X_9 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_10
        self.X_10 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_11
        self.X_11 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_12
        self.X_12 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        # X_13
        self.X_13 = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        self.output = nn.Linear(13*num_neurons, output_size)
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
        student_output["hidden_states"]=[]
        # Interchange intervention
        input_ids = torch.cat([input_ids[0:10], input_ids[12:]]) # Exclude CHO and insulin without scaling
        hooks = []
        layers = [self.X_1,self.X_2,self.X_3,self.X_4,self.X_5,self.X_6,self.X_7,self.X_8,self.X_9,self.X_10,self.X_11,self.X_12, self.X_13]
        for i, layer_module in enumerate(layers):
            if variable_names != None and i in variable_names:
                assert interchanged_variables != None
                for interchanged_variable in variable_names[i]:
                    interchanged_activations = interchanged_variables[interchanged_variable[0]]
                    # #https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#method-3-attach-a-hook AND interchange_with_activation_at()
                    hook = layer_module.register_forward_hook(interchange_hook(interchanged_variable, interchanged_activations))
                    hooks.append(hook)
            x = layer_module(
                input_ids
            )
            student_output["hidden_states"].append(x)
        
        flat_tensor = torch.tensor([item for sublist in student_output["hidden_states"] for item in sublist])
        student_output["outputs"] = self.output(flat_tensor)

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
