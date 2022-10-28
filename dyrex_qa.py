import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForQuestionAnswering
from einops import rearrange, reduce, repeat
from transformers import BertPreTrainedModel, BertModel, RobertaModel
from torch.nn import Module, Linear, LayerNorm, CrossEntropyLoss

class DyRExDecoder(nn.Module):

    def __init__(self, num_layers=3, dim=768, n_head=8, self_attention_type="indep"):
        super().__init__()

        # decoder layers
        layer = nn.TransformerDecoderLayer(dim, n_head, activation="gelu")
        self.base = nn.TransformerDecoder(layer, num_layers=num_layers)

        # self_attention mask
        self.register_buffer('tgt_mask', self._generate_square_subsequent_mask(self_attention_type=self_attention_type))

        # output layer
        self.linear_q = nn.Linear(dim, 2)

    def compute_q(self, x, attention_mask=None):
        batch_size = x.size(0)
        queries = self.linear_q.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # B, 2, dim
        queries = queries.transpose(0, 1)
        x = x.transpose(0, 1)

        if attention_mask is not None:
            attention_mask = attention_mask == 0

        return self.base(queries, x, tgt_mask=self.tgt_mask, memory_key_padding_mask=attention_mask)

    def forward(self, x, attention_mask=None):
        queries = self.compute_q(x, attention_mask)
        start_logits, end_logits = torch.einsum('cbd,bld->cbl', queries, x)
        return start_logits, end_logits

    def _generate_square_subsequent_mask(self, sz=2, self_attention_type="unidirectional"):
        if self_attention_type == "unidirectional":
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        elif self_attention_type == "independent":
            mask = torch.eye(sz)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        elif self_attention_type == "bidirectional":
            mask = torch.zeros(2, 2).float()
        else:
            raise ValueError
        return mask

class DyRexQA(BertPreTrainedModel):

    def __init__(self, config, num_layers=3, self_attention_type="bidirectional"):

        super().__init__(config)

        self.encoder_name = config.model_type
        if "roberta" in self.encoder_name:
            self.bert = RobertaModel(config)
        else:
            self.bert = BertModel(config)

        self.decoder = DyRExDecoder(num_layers=num_layers,
                                    dim=config.hidden_size,
                                    self_attention_type=self_attention_type)

        self.init_weights()

    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None):

        hidden = self.bert.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)[0]

        # start/end logits
        start_logits, end_logits = self.decoder(hidden, attention_mask)

        if attention_mask is not None:
            start_logits = start_logits + (1 - attention_mask) * -10000.0
            end_logits = end_logits + (1 - attention_mask) * -10000.0

        outputs = (start_logits, end_logits)

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions.long())
            end_loss = loss_fct(end_logits, end_positions.long())

            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs