import torch
import torch.nn as nn
from utils.counterfactual_utils import interchange_hook
from utils.loss_fn import RMSELoss

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(22, 13)
        self.fc2 = nn.Linear(13, 10)
        self.loss = RMSELoss()
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")

    # def forward(self, x):
    #     out = self.model(x)
    #     return out

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
        x = input_ids
        hooks = []
        layers = [self.fc1, self.fc2]
        for i, layer_module in enumerate(layers):
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
            student_output["hidden_states"].append(x)
        
        student_output["outputs"] = x[:,0] #glucose level
        
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
                loss = self.loss(s_outputs, t_outputs)
                student_output["loss"] = loss
        else:
            # causal loss.
            causal_s_outputs = student_output["outputs"]
            loss = self.loss(causal_s_outputs, causal_t_outputs)
            student_output["loss"] = loss

            # measure the efficacy of the interchange.
            teacher_interchange_efficacy = (
                self.ce_loss_fct(
                    nn.functional.log_softmax(causal_t_outputs, dim=-1),
                    nn.functional.softmax(t_outputs, dim=-1),
                )
            )

            student_interchange_efficacy = (
                self.ce_loss_fct(
                    nn.functional.log_softmax(causal_s_outputs, dim=-1),
                    nn.functional.softmax(s_outputs, dim=-1),
                )
            )
            student_output["teacher_interchange_efficacy"] = teacher_interchange_efficacy
            student_output["student_interchange_efficacy"] = student_interchange_efficacy
        
        for h in hooks:
            h.remove()

        return student_output
