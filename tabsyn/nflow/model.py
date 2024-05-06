import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor
from tabsyn.nflow.flows import NormalizingFlows

import typing as ty
import math

class Tokenizer(nn.Module):

    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
    
        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]

        return x




class Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))  
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num  = h[:, :self.d_numerical]
        h_cat  = h[:, self.d_numerical:]

        #print('hnum_shape = ', h_num.shape)
        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
      
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat
        
class AE(nn.Module): 
    def __init__(self, d_numerical, categories, d_token, bias = True):
        super(AE, self).__init__()
        
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def load_weights(self, Pretrained_AE):
        self.Tokenizer.load_state_dict(Pretrained_AE.Tokenizer.state_dict())
        self.Detokenizer.load_state_dict(Pretrained_AE.Reconstructor.state_dict())        
        
    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)

        recon_x_num, recon_x_cat = self.Reconstructor(x)

        return recon_x_num, recon_x_cat


class Model_NFLOW_nextgen(nn.Module):
    def __init__(self, d_numerical, categories, d_token, bias = True):
        super(Model_NFLOW_nextgen, self).__init__()
        
        d_bias = d_numerical + len(categories)
        self.features = d_bias*d_token
         
        self.nflow = NormalizingFlows(n_features=self.features)        

    def forward(self, count: int) -> torch.Tensor:
        self.flow.eval()
        with torch.no_grad():
            return self.flow.sample(count)
            
    def log_prob(self, inputs):
        return self.nflow.flow.log_prob(inputs)
        
class Encoder_NFLOW_model_nextgen(nn.Module):
    def __init__(self, d_numerical, categories, d_token, bias = True):
        d_bias = d_numerical + len(categories)
        self.features = d_bias*d_token
        
        super(Encoder_NFLOW_model_nextgen, self).__init__()
        #self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.nflow = NormalizingFlows(n_features=self.features) 

    def load_weights(self, Pretrained_NFLOW_nextgen):
        #self.Tokenizer.load_state_dict(Pretrained_AE.Tokenizer.state_dict())
        self.nflow.load_state_dict(Pretrained_NFLOW_nextgen.nflow.state_dict())

    def forward(self, inputs):
        #x = self.Tokenizer(x_num, x_cat)
        #x = x.view(-1, self.features)
        z = self.nflow.flow.transform_to_noise(inputs)

        return z
        
class Decoder_NFLOW_model_nextgen(nn.Module):
    def __init__(self, d_numerical, categories, d_token, bias = True):
        super(Decoder_NFLOW_model_nextgen, self).__init__()
        d_bias = d_numerical + len(categories)
        self.features = d_bias*d_token
        self.d_token = d_token

        self.nflow = NormalizingFlows(n_features=self.features) 
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)
        
    def load_weights(self, Pretrained_NFLOW_nextgen, Reconstructor):
        self.nflow.load_state_dict(Pretrained_NFLOW_nextgen.nflow.state_dict())
        self.Detokenizer.load_state_dict(Reconstructor.state_dict())

    def forward(self, z):

        h, _ = self.nflow.flow._transform.inverse(z)
        h = h.view(h.shape[0], int(self.features/self.d_token), self.d_token)
        x_hat_num, x_hat_cat = self.Detokenizer(h)

        return x_hat_num, x_hat_cat