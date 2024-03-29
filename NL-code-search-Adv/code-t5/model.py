# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, code_inputs,nl_inputs, decoder_input_ids=None,return_vec=False): 
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        
        if decoder_input_ids is None:
            # 如果没有提供decoder_input_ids，我们需要创建它们
            # 例如，使用与nl_inputs相同的尺寸，但值全为特定的起始令牌ID
            decoder_input_ids = torch.full_like(nl_inputs, self.tokenizer.pad_token_id)
        
        # 注意，我们假设encoder具有编解码器的结构
        # 在这个假设下，你可能需要根据你的模型类型调整forward的调用
        encoded_outputs = self.encoder(
            input_ids=inputs,
            attention_mask=inputs.ne(self.tokenizer.pad_token_id),
            decoder_input_ids=decoder_input_ids
        ) # 假设输出的第二个元素是我们需要的向量表示

        
        last_hidden_states = encoded_outputs.encoder_last_hidden_state
        outputs = last_hidden_states.mean(dim=1)

        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        
        if return_vec:
            return code_vec, nl_vec

        
        scores = (nl_vec[:, None, :] * code_vec[None, :, :]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        
        return loss, code_vec, nl_vec

      
        
 
