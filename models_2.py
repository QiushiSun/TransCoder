import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5ForConditionalGeneration, BartForConditionalGeneration, AutoModelForSeq2SeqLM, RobertaConfig, RobertaModel, RobertaTokenizer

import logging
import sys
from GAT_prefix import CodeGraphPrefix
#import codecs
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
logger = logging.getLogger(__name__)

MODEL_CHECKPOINTS = {'roberta': 'roberta-base',
                     'codebert': 'microsoft/codebert-base',
                     'graphcodebert': 'microsoft/graphcodebert-base',
                     't5': 't5-base',
                     'codet5': 'Salesforce/codet5-base',
                     'bart': 'facebook/bart-base',
                     'plbart': 'uclanlp/plbart-base',
                     'unixcoder':'microsoft/unixcoder-base'}


MODEL_LOCALS = {
    'roberta': 'roberta-base',
    'codebert': 'codebert-base',
    'graphcodebert': 'graphcodebert-base',
    't5': 't5-base',
    'codet5':'codet5-base',
    'bart': 'bart-base',
    'plbart': 'plbart-base',
    'unixcoder':'unixcoder-base',
}
MODEL_CLASSES = {'roberta': (AutoConfig, AutoModel, AutoTokenizer),
                 'codebert': (AutoConfig, AutoModel, AutoTokenizer),
                 'graphcodebert': (AutoConfig, AutoModel, AutoTokenizer),
                 'unixcoder':(AutoConfig, AutoModel, AutoTokenizer),
                 't5': (AutoConfig, T5ForConditionalGeneration, AutoTokenizer),
                 'codet5': (AutoConfig, T5ForConditionalGeneration, AutoTokenizer),
                 'bart': (AutoConfig, BartForConditionalGeneration, AutoTokenizer),
                 'plbart':(AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer)}

# model_name_to_backbone = {
#     "roberta": "roberta",
#     "codebert": "roberta",
#     "graphcodebert": "roberta",
#     "unixcoder": "roberta",
#     "t5": "t5",
#     "codet5": "t5",
#     "bart": "bart",
#     "plbart": "bart",
# }

# MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
#                  't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
#                  'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
#                  'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def bulid_or_load_gen_model(args):
    # checkpoint = MODEL_CHECKPOINTS[args.model_name]
    checkpoint = os.path.join(args.huggingface_locals, MODEL_LOCALS[args.model_name])
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
    config = config_class.from_pretrained(checkpoint)
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    
    if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
        encoder = model_class.from_pretrained(checkpoint, output_attentions=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(encoder=encoder, decoder=decoder,
                        config=config, beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    elif args.model_name in ['unixcoder']:
        # import！！！you must set is_decoder as True for generation in unixcoder！！！
        config.is_decoder = True
        encoder = model_class.from_pretrained(checkpoint, config=config)
        if args.task in ['complete']:
            if args.sub_task == "python":
                eos_ids = [tokenizer.sep_token_id]
            else:
                eos_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'), tokenizer.convert_tokens_to_ids('Ġ{')]
            model=Seq2Seq4UniXcoder_completion(encoder=encoder,decoder=encoder,config=config,
                        beam_size=args.beam_size,max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id,eos_id=eos_ids)
        elif args.task in ['generate']:
            model = Seq2Seq4UniXcoder_generation(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
        elif args.task in ['summarize','translate','refine']:
            model = Seq2Seq4UniXcoder_e2d(encoder=encoder,decoder=encoder,config=config,
                        beam_size=args.beam_size,max_length=args.max_target_length,
                        sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
            
    elif args.model_name in ['t5', 'codet5','bart','plbart']:
        model = model_class.from_pretrained(checkpoint, output_attentions=True)

    logger.info("Finish loading model [%s] parameters from %s", get_model_size(model), args.model_name)

    return config, model, tokenizer

def bulid_or_load_cls_model(args):
    # checkpoint = MODEL_CHECKPOINTS[args.model_name]
    checkpoint = os.path.join(args.huggingface_locals, MODEL_LOCALS[args.model_name])
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
    config = config_class.from_pretrained(checkpoint)
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    # if args.model_name in ['unixcoder']:
    #     model = model_class.from_pretrained(checkpoint, output_attentions=True)
    #     model = Model4UniXcoder(model,config,tokenizer,args)

    model = model_class.from_pretrained(checkpoint, output_attentions=True)
    
    if args.task == 'defect':
        model = DefectModel(model, config, tokenizer, args)
    elif args.task == 'clone':
        # model.resize_token_embeddings(32000)
        model = CloneModel(model, config, tokenizer, args)

    logger.info("Finish loading model [%s] parameters from %s", get_model_size(
        model), args.model_name)

    return config, model, tokenizer

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model4UniXcoder(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model4UniXcoder, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids = input_ids.view(-1,self.args.max_source_length)
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        outputs = (outputs * input_ids.ne(1)[:,:,None]).sum(1)/input_ids.ne(1).sum(1)[:,None]
        outputs = outputs.reshape(-1,2,outputs.size(-1))
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        if labels is not None:
            loss = ((cos_sim-labels.float())**2).mean()
            return loss,cos_sim
        else:
            return cos_sim

class CloneModel(nn.Module):
    def __init__(self, encoder: AutoModel, config, tokenizer, args):
        super(CloneModel, self).__init__()
        checkpoint = os.path.join(args.huggingface_locals, MODEL_LOCALS[args.model_name])
        config = AutoConfig.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encoder = encoder # backbone
        # backbone_type = model_name_to_backbone[args.model_name]
        # embeddings_weight = None
        # if backbone_type == 'roberta':
        #     embeddings_weight = self.encoder.embeddings.word_embeddings.weight
        # elif backbone_type == 't5':
        #     embeddings_weight = self.encoder.embeddings.word_embeddings.weight
        # elif backbone_type == 'bart':
        #     embeddings_weight = self.encoder.embeddings.word_embeddings.weight
        embeddings_weight = self.encoder.embeddings.word_embeddings.weight
        
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        # ============ add ==========
        for param in self.encoder.parameters():
            param.requires_grad = False   #bert的参数是定死的   这里在code里加进来
        # load retrieved prefix code tokens
        self.code_prefix_tokens = torch.Tensor([10243, 1045, 34, 1003, 53,222]).long().cuda()
        self.code_prefix_matrix = torch.ones(6, 6).cuda()
        self.pre_seq_len = len(self.code_prefix_tokens) # 5
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        # add prefix encoder
        self.code_prefix = CodeGraphPrefix(self.config, embeddings_weight)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # ========================  ==
    
    # ===== add ====
    def get_prompt(self, batch_size):
        code_prefix_tokens = self.code_prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.encoder.device)
        code_prefix_matrix = self.code_prefix_matrix.unsqueeze(0).expand(batch_size, -1, -1).to(self.encoder.device)
        past_key_values = self.code_prefix(code_prefix_tokens, code_prefix_matrix)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    # =================

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        # ============= add  ===================
        batch_size = attention_mask.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size) # add
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.encoder.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask.float()), dim=1)
        # ======================================
        outputs = self.encoder(
            input_ids=source_ids, 
            attention_mask=attention_mask,
            labels=source_ids, 
            decoder_attention_mask=attention_mask, 
            output_hidden_states=True, 
            past_key_values=past_key_values # add
            )
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        # ============= add  ===================
        batch_size = attention_mask.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size) # add
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.encoder.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask.float()), dim=1)
        # ======================================
        outputs = self.encoder(
            input_ids=source_ids, 
            attention_mask=attention_mask,
            labels=source_ids, 
            decoder_attention_mask=attention_mask, 
            output_hidden_states=True,
            past_key_values=past_key_values # add
            )
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        # ============= add  ===================
        batch_size = attention_mask.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size) # shape: (2, 6, 12, 6, 64)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.encoder.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask.float()), dim=1)
        # ======================================
        vec = self.encoder(
            input_ids=source_ids, 
            attention_mask=attention_mask, 
            past_key_values=past_key_values # add
            )[0][:, 0, :]
        return vec

    def get_unixcoder_vec(self, source_ids):
        attention_mask = source_ids.ne(1)
        # ============= add  ===================
        batch_size = attention_mask.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size) # add
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.encoder.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask.float()), dim=1)
        # ======================================
        # outputs = self.encoder(source_ids,attention_mask=source_ids.ne(1))[0]#shape:batch_size*max_len512*hidden_size768
        outputs = self.encoder(
            source_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values # add
        )[0]#shape:batch_size*max_len512*hidden_size768
        outputs = (outputs * source_ids.ne(1)[:,:,None]).sum(1)/source_ids.ne(1).sum(1)[:,None]#shape:batch_size*hidden_size
        outputs = outputs.reshape(-1,2,outputs.size(-1))#shape:batch_size/2 *2*hidden_size
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        return cos_sim #cos_sim, labels
    

        
    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)#[batch*2,512]

        if self.args.model_name in ['t5','codet5']:
            vec = self.get_t5_vec(source_ids)#[batch*2,768]
            logits = self.classifier(vec)#[batch,2]
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['bart','plbart']:
            vec = self.get_bart_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['roberta', 'codebert', 'graphcodebert']:
            vec = self.get_roberta_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['unixcoder']:
            logits = self.get_unixcoder_vec(source_ids)
            prob = logits #=cos_sim

        if labels is not None:
            if self.args.model_name not in ['unixcoder']:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                loss = ((logits-labels.float())**2).mean()
                return loss, prob
        else:
            return prob


class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def get_unixcoder_vec(self, source_ids):
        outputs = self.encoder(source_ids,attention_mask=source_ids.ne(1))[0]#shape:batch_size*max_len512*hidden_size768
        outputs = (outputs * source_ids.ne(1)[:,:,None]).sum(1)/source_ids.ne(1).sum(1)[:,None]#shape:batch_size*hidden_size
        # outputs = outputs.reshape(-1,2,outputs.size(-1))
        # outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        # cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        # return cos_sim #cos_sim, labels
        outputs = self.classifier(outputs)
        return outputs
    
    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        if self.args.model_name in ['t5','codet5']:
            vec = self.get_t5_vec(source_ids)#[batch_size,hidden_size]
            logits = self.classifier(vec)#[batch_size,2]
            prob = nn.functional.softmax(logits)#[batch_size,2]
        elif self.args.model_name in ['bart','plbart']:
            vec = self.get_bart_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['roberta', 'codebert', 'graphcodebert']:
            vec = self.get_roberta_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['unixcoder']:
            logits = self.get_unixcoder_vec(source_ids)
            prob = logits #=cos_sim


        if labels is not None:
            # if self.args.model_name not in ['unixcoder']:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)#[batchsize,2] [batchsize]
            return loss, prob
            # else:
            #     loss = ((logits-labels.float())**2).mean()
            #     return loss, prob
        else:
            return prob



class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)#source_mask size: [batch_size, source_length=256]
        encoder_attention = outputs[-1]#[batch, 256, 768]
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()#[256, batch, 768]
        if target_ids is not None:
            attn_mask = -1e4 * \
                (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])#[128,128] upper triangular=-10000 lower=0 mask upper
            tgt_embeddings = self.encoder.embeddings(
                target_ids).permute([1, 0, 2]).contiguous()#[128, batch, 768]
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=~source_mask)#[128, batch, 768]
            # memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(
                out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
            return loss, loss * active_loss.sum(), active_loss.sum(), encoder_attention
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * \
                        (1 - self.bias[:input_ids.shape[1],
                         :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(
                        input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=~context_mask)
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute(
                        [1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(
                        0, beam.getCurrentOrigin()))
                    input_ids = torch.cat(
                        (input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds, encoder_attention

class Seq2Seq4UniXcoder_e2d(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq4UniXcoder_e2d, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id       
        
    def forward(self, source_ids, target_ids=None):   
        if target_ids is None:
            return self.generate(source_ids)
        
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        #[batch,256,256] case: upper left 70*70(source) true other false
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)
        ids = torch.cat((source_ids,target_ids),-1)
        #[batch,384] case: total source70 not 1 and target 15 not 1=85
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        #[batch,256:384,0:384]=[batch,128,384],upper left 384*256 true,lower right 128*128 lower triangle 
        mask = mask & ids[:,None,:].ne(1)
        #[batch,128,384] set redundance 1 to false
        out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = loss,loss*active_loss.sum(),active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)        
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break

                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   

class Seq2Seq4UniXcoder_completion(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq4UniXcoder_completion, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight=self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids,train=False): 
        max_length = source_ids.ne(1).sum(-1).max()
        source_ids = source_ids[:,:max_length]        
        if train:  
            length = source_ids.size(-1)
            out = self.decoder(source_ids,attention_mask=self.bias[:,:length,:length]).last_hidden_state
            lm_logits = self.lm_head(out)
            # Shift so that tokens < n predict n
            active_loss = source_ids[..., 1:].ne(1).view(-1)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = source_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            #Predict 
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)   
            source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
            length = source_ids.size(-1)
            encoder_output = self.decoder(source_ids,attention_mask=self.bias[:,:length,:length])
            for i in range(source_ids.shape[0]):
                context=[[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                         for y in encoder_output.past_key_values]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState()
                context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
                out = encoder_output.last_hidden_state[i:i+1,:source_len[i]].repeat(self.beam_size,1,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    if _ == 0: 
                        hidden_states=out[:,-1,:]
                        out = self.lsm(self.lm_head(hidden_states)).data
                        beam.advance(out)
                        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                        input_ids=beam.getCurrentState()
                    else:
                        length = context_ids.size(-1)+input_ids.size(-1)
                        out = self.decoder(input_ids,attention_mask=self.bias[:,context_ids.size(-1):length,:length],
                                           past_key_values=context).last_hidden_state
                        hidden_states=out[:,-1,:]
                        out = self.lsm(self.lm_head(hidden_states)).data
                        beam.advance(out)
                        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                        input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)    

            return preds   

class Seq2Seq4UniXcoder_generation(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq4UniXcoder_generation, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id       
        
    def forward(self, source_ids, target_ids=None):   
        if target_ids is None:
            return self.generate(source_ids)
        
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)  
        ids = torch.cat((source_ids,target_ids),-1)
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        mask = mask & ids[:,None,:].ne(1)

        out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = loss,loss*active_loss.sum(),active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)        
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break

                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size] #len:10
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   
        
        

class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
