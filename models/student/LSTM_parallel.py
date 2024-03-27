import torch
import torch.nn as nn
from utils.counterfactual_utils import interchange_hook

class LSTM_parallel(nn.Module):
    '''
        The LSTM model is based on the Long Short-Term Memory Recurrent Neural Network.
        Parameters:
            - n_in: number of features per time-step
            - n_out: number of outputs (shall be 1)
            - n_hidden: number of hidden layers
            - n_neurons: number of neurons per hidden layer
            - loss: loss function used (shall be "MSE")
            - dropout: dropout coefficient
            - l2: L2 penalty applied to the weights
            - epochs: maximal number of training epochs
            - batch_size: size of the mini-batch
            - lr: initial learning rate
            - patience: number of non-improving epochs after which we early stop the training
    '''
    def __init__(self, n_in=20, n_out=1, n_hidden=256, n_neurons=1, dropout=0.3):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons #layers
        self.dropout = dropout

        # batch_first=True => input/ouput w/ shape (batch,seq,feature)
        self.lstm_1= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_2= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_3= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_4= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_4= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_5= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_6= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_7= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_8= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_9= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_10= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_11= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_12= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.lstm_13= nn.LSTM(self.n_in,
                            self.n_hidden,
                            self.n_neurons,
                            bidirectional=False,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout)

        self.output = nn.Linear(13*n_hidden, n_out)
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
        lstm_outputs = []
        # Interchange intervention
        input_ids = torch.cat([input_ids[0:10], input_ids[12:]]).unsqueeze(0) # Exclude CHO and insulin without scaling

        hooks = []
        layers = [self.lstm_1,self.lstm_2,self.lstm_3,self.lstm_4,self.lstm_5,self.lstm_6,self.lstm_7,self.lstm_8,self.lstm_9,self.lstm_10,self.lstm_11,self.lstm_12, self.lstm_13]
        for i, layer_module in enumerate(layers):
            if variable_names != None and i in variable_names:
                assert interchanged_variables != None
                for interchanged_variable in variable_names[i]:
                    interchanged_activations = interchanged_variables[interchanged_variable[0]]
                    # #https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#method-3-attach-a-hook AND interchange_with_activation_at()
                    hook = layer_module.register_forward_hook(interchange_hook(interchanged_variable, interchanged_activations))
                    hooks.append(hook)
            x, (h, _) = layer_module(
                input_ids
            )
            student_output["hidden_states"].append(h)
            lstm_outputs.append(x.squeeze(0))
        
        flat_tensor = torch.tensor([item for sublist in lstm_outputs for item in sublist])
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