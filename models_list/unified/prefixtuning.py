#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import AutoTokenizer
from .base import PushToHubFriendlyModel
# from ..prompt.modeling_auto import AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM
from utils import get_graph_metadata
from transformers.modeling_outputs import Seq2SeqLMOutput
class E2D_Model_Prefix(PushToHubFriendlyModel):
    def __init__(self, args, shared_state_dict_list):
        super().__init__()
        self.args = args

        """The prefix-tuning code"""

        self.preseqlen = args.max_source_length
        self.mid_dim = args.gat_token_num




        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, use_fast=False)
        
        # self.code_prefix_tokens, self.code_prefix_matrix = get_graph_metadata(self.args,self.tokenizer)
        self.code_prefix_tokens = torch.arange(self.mid_dim).long().to(args.device)
        self.code_prefix_matrix = torch.arange(self.mid_dim).long().to(args.device)
        
        # self.code_prefix_matrix = torch.tensor(self.code_prefix_matrix, dtype=torch.long).cuda()
        # self.code_prefix_tokens = self.code_prefix_tokens.unsqueeze(0).expand(self.args.batch_size, -1)
        # self.code_prefix_matrix = self.code_prefix_matrix.unsqueeze(0).expand(self.args.batch_size, -1, -1)
        from ..prompt.modeling_bart import BartForConditionalGeneration
        from ..prompt.modeling_t5 import T5ForConditionalGeneration
        if "bart" in self.args.pretrained_model_name_or_path:
            self.pretrain_model = BartForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path
            )
            embeddings_weight = self.pretrain_model.model.shared.weight
            self.model= self.pretrain_model.model # add
            self.config = self.pretrain_model.config
            self.match_n_layer = self.config.decoder_layers
            self.match_n_head = self.config.decoder_attention_heads
            self.n_embd = self.config.d_model
            assert self.n_embd % self.match_n_head == 0
            self.match_n_embd = self.n_embd // self.match_n_head # huggingface BART's dim of kv need to be calculated
            assert isinstance(self.pretrain_model, (BartForConditionalGeneration))
        elif "t5" in self.args.pretrained_model_name_or_path:
            self.pretrain_model = T5ForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path
            )
            self.shared = self.pretrain_model.shared # add
            embeddings_weight = self.pretrain_model.shared.weight
            self.config = self.pretrain_model.config
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
            assert isinstance(self.pretrain_model, (T5ForConditionalGeneration))
        else:
            print(self.pretrain_model)
            raise ValueError("Other models are not supported yet!")

        # if args.special_tokens:
        #     self.tokenizer.add_tokens([v for k, v in args.special_tokens])
        #     self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        if self.args.knowledge_usage == 'separate':
            if self.args.prefix_tuning =='pass_tuning':
                from GAT_prefix import CodeGraphPrefix
                self.knowledge_trans = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            elif self.args.prefix_tuning =='GCN':
                from GCN_prefix import CodeGraphPrefix
                self.knowledge_trans = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            elif self.args.prefix_tuning == 'prefix_tuning':
                self.knowledge_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
                )
                if len(shared_state_dict_list)==3:
                    self.knowledge_trans.load_state_dict(shared_state_dict_list[0])

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        if self.args.knowledge_usage == 'separate':
            if self.args.prefix_tuning =='pass_tuning':
                from GAT_prefix import CodeGraphPrefix
                self.knowledge_trans_enc = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            elif self.args.prefix_tuning =='GCN':
                from GCN_prefix import CodeGraphPrefix
                self.knowledge_trans_enc = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            elif self.args.prefix_tuning == 'prefix_tuning':
                self.knowledge_trans_enc = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
                )
                if len(shared_state_dict_list)==3:
                    self.knowledge_trans.load_state_dict(shared_state_dict_list[1])

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        # Knowledge prompt.
        if self.args.knowledge_usage == 'separate':
            if self.args.prefix_tuning =='pass_tuning':
                from GAT_prefix import CodeGraphPrefix
                self.knowledge_trans_dec = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            elif self.args.prefix_tuning =='GCN':
                from GCN_prefix import CodeGraphPrefix
                self.knowledge_trans_dec = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            elif self.args.prefix_tuning == 'prefix_tuning':
                self.knowledge_trans_dec = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
                )
                if len(shared_state_dict_list)==3:
                    self.knowledge_trans.load_state_dict(shared_state_dict_list[2])

        self.dropout = nn.Dropout(args.prefix_dropout)
        import numpy as np
        if self.args.fix_model_param:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        if 1 or not self.args.prefix_tuning:
            for param in self.wte.parameters():
                param.requires_grad = False
            for param in self.control_trans.parameters():
                param.requires_grad = False
            for param in self.wte_dec.parameters():
                param.requires_grad = False
            for param in self.control_trans_dec.parameters():
                param.requires_grad = False
            for param in self.wte_enc.parameters():
                param.requires_grad = False
            for param in self.control_trans_enc.parameters():
                param.requires_grad = False

    # @classmethod
    # def from_pretrained(self,pretrained_model_name_or_path,**kwargs):
    #     args=kwargs['args']
    #     args.pretrained_model_name_or_path= pretrained_model_name_or_path
    #     self.__init__(self,args=kwargs['args'])
    #     return self
    
    def get_prompt(self, bsz=None, sample_size=1, description=None, knowledge=None, knowledge_matrix=None):
        old_bsz = bsz
        bsz = bsz * sample_size
        #print("input_tokens", self.input_tokens)#tensor([  0,   1,   2,   3,..255])
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        if description is not None:
            temp_control = temp_control + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb


        # print('knowledge.shape',knowledge.shape)
        # print('knowledge_matrix.shape',knowledge_matrix.shape)
        knowledge = knowledge.unsqueeze(0).expand(old_bsz, -1)
        knowledge_matrix = knowledge_matrix.unsqueeze(0).expand(old_bsz, -1, -1)
        if self.args.adjcency_mode=='fully-connected':
            knowledge_matrix = torch.where(knowledge_matrix >0,  torch.ones_like(knowledge_matrix), torch.zeros_like(knowledge_matrix))
        elif self.args.adjcency_mode=='sast':
            knowledge_matrix = torch.where(knowledge_matrix ==1, torch.ones_like(knowledge_matrix), torch.zeros_like(knowledge_matrix)) 
        # print('knowledge.shape',knowledge.shape)
        # print('knowledge_matrix.shape',knowledge_matrix.shape)

        if knowledge is not None:
            if self.args.prefix_tuning in ['pass_tuning','GCN']:
        
                past_key_values = torch.cat([past_key_values, self.knowledge_trans(
                    knowledge.repeat_interleave(sample_size, dim=0),
                    knowledge_matrix.repeat_interleave(sample_size, dim=0))], dim=1)#注意.repeat_interleave(sample_size, dim=0)
            elif self.args.prefix_tuning == 'prefix_tuning':
                past_key_values = torch.cat([past_key_values, self.knowledge_trans(
                    knowledge.repeat_interleave(sample_size, dim=0),)], dim=1)# Encode knowledge.



        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        if description is not None:
            temp_control_dec = temp_control_dec + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            if self.args.prefix_tuning in ['pass_tuning','GCN']:
                past_key_values_dec = torch.cat([past_key_values_dec, self.knowledge_trans_dec(
                    knowledge.repeat_interleave(sample_size, dim=0),
                    knowledge_matrix.repeat_interleave(sample_size, dim=0))], dim=1)#注意.repeat_interleave(sample_size, dim=0)
            elif self.args.prefix_tuning == 'prefix_tuning':
                past_key_values_dec = torch.cat([past_key_values_dec, self.knowledge_trans_dec(
                    self.tokenizer.encode(knowledge.replace('</s>', '<unk>'), 
                    max_length=self.args.max_source_length, padding='max_length', truncation=True))], dim=1)
        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        # print("input_tokens_enc.shape = ", input_tokens_enc.shape)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        # print("temp_control_enc.shape = ", temp_control_enc.shape)
        if description is not None:
            temp_control_enc = temp_control_enc + description.unsqueeze(1)
        # print("temp_control_enc.shape = ", temp_control_enc.shape)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            # print("past_key_values_enc.shape = ", past_key_values_enc.shape)
            # print("self.knowledge_trans_enc(knowledge,knowledge_matrix).shape = ", self.knowledge_trans_enc(knowledge, knowledge_matrix).shape)
            if self.args.prefix_tuning in ['pass_tuning','GCN']:
                past_key_values_enc = torch.cat([past_key_values_enc, self.knowledge_trans_enc(
                    knowledge,knowledge_matrix)], dim=1)
            elif self.args.prefix_tuning == 'prefix_tuning':
                past_key_values_enc = torch.cat([past_key_values_enc, self.knowledge_trans_enc(
                    self.tokenizer.encode(knowledge.replace('</s>', '<unk>'), 
                    max_length=self.args.max_source_length, padding='max_length', truncation=True))], dim=1)

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result

    def get_origin_prompt(self, bsz=None, sample_size=1, description=None, knowledge=None):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        knowledge = knowledge.unsqueeze(0).expand(old_bsz, -1, -1)
        temp_control = self.wte(input_tokens)
        if description is not None:
            temp_control = temp_control + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        # print("old_bsz = ", old_bsz)
        # print("sample_size = ", sample_size)
        # print("bsz = ", bsz)
        # print("knowledge.shape = ", knowledge.shape)
        # print("past_key_values.shape = ", past_key_values.shape)
        if knowledge is not None:
            past_key_values = torch.cat([past_key_values, self.knowledge_trans(knowledge.repeat_interleave(sample_size, dim=0))], dim=1)
        #[bsz, seqlen 256, layer*emb 18432] self.n_embd, self.mid_dim
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        if description is not None:
            temp_control_dec = temp_control_dec + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_dec = torch.cat([past_key_values_dec, self.knowledge_trans_dec(knowledge.repeat_interleave(sample_size, dim=0))], dim=1)

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)
        if description is not None:
            temp_control_enc = temp_control_enc + description.unsqueeze(1)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_enc = torch.cat([past_key_values_enc, self.knowledge_trans_enc(knowledge)], dim=1)

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result
    def get_description_representation(self, kwargs):
        if self.args.use_description and self.args.map_description:
            description_input_ids = kwargs.pop("description_input_ids")
            description_attention_mask = kwargs.pop("description_attention_mask")
            if "t5" in self.args.pretrained_model_name_or_path:
                description_outputs = self.pretrain_model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            elif "bart" in self.args.pretrained_model_name_or_path:
                description_outputs = self.pretrain_model.model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            else:
                raise ValueError()
        else:
            description = None

        return description

    def get_knowledge_representation(self, kwargs):
        if self.args.knowledge_usage == 'separate':
            knowledge_input_ids = kwargs.pop("knowledge_input_ids", None)
            knowledge_matrix = kwargs.pop("knowledge_matrix", None)
            if "t5" in self.args.pretrained_model_name_or_path:
                knowledge_outputs = self.pretrain_model.encoder(
                    input_ids=knowledge_input_ids,
                    # attention_mask=knowledge_matrix,
                )
                knowledge = knowledge_outputs.last_hidden_state #shape: (bsz, seqlen, dim)
                if knowledge.shape[0] == 1:
                    knowledge = knowledge[0]
                    # print('knowledge.shape = ', knowledge.shape)
            elif "bart" in self.args.pretrained_model_name_or_path:
                knowledge_outputs = self.pretrain_model.model.encoder(
                    input_ids=knowledge_input_ids,
                    # attention_mask=knowledge_matrix,
                )
                knowledge = knowledge_outputs.last_hidden_state
                if knowledge.shape[0] == 1:
                    knowledge = knowledge[0]
            else:
                raise ValueError()
        elif self.args.knowledge_usage == 'concatenate':
            knowledge = None
        else:
            raise ValueError()

        return knowledge

    def forward(self,
                input_ids,
                labels,
                attention_mask = None,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        kwargs["knowledge_input_ids"] = self.code_prefix_tokens
        kwargs["knowledge_matrix"] = self.code_prefix_matrix

        # # Encode description.
        # description_representation = self.get_description_representation(kwargs)

        # # Encode knowledge.
        # knowledge_representation = self.get_knowledge_representation(kwargs)

        # past_prompt = self.get_prompt(
        #     bsz=bsz, description=description_representation, knowledge=knowledge_representation,
        # ) #past_prompt shape: obj:`(batch_size, num_beams, num_layers, num_heads, sequence_length, embed_size_per_head)`

        if self.args.prefix_tuning in ['pass_tuning','GCN']:
            past_prompt = self.get_prompt(
                bsz=bsz, knowledge=kwargs["knowledge_input_ids"],knowledge_matrix=kwargs["knowledge_matrix"])
        elif self.args.prefix_tuning == 'prefix_tuning':
            # Encode knowledge.
            knowledge_representation = self.get_knowledge_representation(kwargs)

            past_prompt = self.get_origin_prompt(
                bsz=bsz, knowledge=knowledge_representation,
            )
        #past_prompt shape: obj:`(batch_size, num_beams, num_layers, num_heads, sequence_length, embed_size_per_head)`

        if attention_mask == None:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # loss = self.pretrain_model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels,
        #     past_prompt=past_prompt,
        #     output_hidden_states=True,
        # ).loss
        # return Seq2SeqLMOutput(loss=loss)
        return self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
            output_hidden_states=True,
        )# without using decoder!!!
        # return {'loss': loss}

    def generate(self,
                 input_ids,
                 attention_mask,
                 use_cache=False,
                 **kwargs):

        bsz = input_ids.shape[0]

        if self.args.prefix_tuning == 'prefix_tuning':
            kwargs["knowledge_input_ids"] = self.code_prefix_tokens
            kwargs["knowledge_matrix"] = self.code_prefix_matrix

        # # Encode description.
        # description_representation = self.get_description_representation(kwargs)

        # # Encode knowledge.
        # knowledge_representation = self.get_knowledge_representation(kwargs)

        # past_prompt = self.get_prompt(
        #     bsz=bsz, sample_size=kwargs['num_beams'], description=description_representation, knowledge=knowledge_representation,
        # )
        if self.args.prefix_tuning in ['pass_tuning','GCN']:
            past_prompt = self.get_prompt(
                bsz=bsz, sample_size=kwargs['num_beams'], knowledge=self.code_prefix_tokens,knowledge_matrix=self.code_prefix_matrix)
        elif self.args.prefix_tuning == 'prefix_tuning':
            # Encode knowledge.
            knowledge_representation = self.get_knowledge_representation(kwargs)

            past_prompt = self.get_origin_prompt(
                bsz=bsz, sample_size=kwargs['num_beams'], knowledge=knowledge_representation,
            )

        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=use_cache,
            **kwargs,
        )

        return generated_ids