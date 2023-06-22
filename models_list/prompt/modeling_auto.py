# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auto Model class. """


import warnings
from collections import OrderedDict

from transformers.utils import logging

# Add modeling imports here
from transformers.models.albert.modeling_albert import (
    AlbertForMaskedLM,
    AlbertForMultipleChoice,
    AlbertForPreTraining,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertModel,
)
# Instead of loading the BART from the transformers==4.9.1, we choose to load from our own prefix-tuning version.
from .modeling_bart import (
    BartForConditionalGeneration,
    BartModel,
)
from transformers.models.bart.modeling_bart import (
    BartForCausalLM,
    BartForQuestionAnswering,
    BartForSequenceClassification,
)
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLMHeadModel,
    BertModel,
)
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationDecoder, BertGenerationEncoder
from transformers.models.big_bird.modeling_big_bird import (
    BigBirdForCausalLM,
    BigBirdForMaskedLM,
    BigBirdForMultipleChoice,
    BigBirdForPreTraining,
    BigBirdForQuestionAnswering,
    BigBirdForSequenceClassification,
    BigBirdForTokenClassification,
    BigBirdModel,
)
from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import (
    BigBirdPegasusForCausalLM,
    BigBirdPegasusForConditionalGeneration,
    BigBirdPegasusForQuestionAnswering,
    BigBirdPegasusForSequenceClassification,
    BigBirdPegasusModel,
)
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotForCausalLM, BlenderbotForConditionalGeneration, BlenderbotModel
from transformers.models.blenderbot_small.modeling_blenderbot_small import (
    BlenderbotSmallForCausalLM,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallModel,
)
from transformers.models.camembert.modeling_camembert import (
    CamembertForCausalLM,
    CamembertForMaskedLM,
    CamembertForMultipleChoice,
    CamembertForQuestionAnswering,
    CamembertForSequenceClassification,
    CamembertForTokenClassification,
    CamembertModel,
)
from transformers.models.canine.modeling_canine import (
    CanineForMultipleChoice,
    CanineForQuestionAnswering,
    CanineForSequenceClassification,
    CanineForTokenClassification,
    CanineModel,
)
from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.convbert.modeling_convbert import (
    ConvBertForMaskedLM,
    ConvBertForMultipleChoice,
    ConvBertForQuestionAnswering,
    ConvBertForSequenceClassification,
    ConvBertForTokenClassification,
    ConvBertModel,
)
from transformers.models.ctrl.modeling_ctrl import CTRLForSequenceClassification, CTRLLMHeadModel, CTRLModel
from transformers.models.deberta.modeling_deberta import (
    DebertaForMaskedLM,
    DebertaForQuestionAnswering,
    DebertaForSequenceClassification,
    DebertaForTokenClassification,
    DebertaModel,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2ForMaskedLM,
    DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    DebertaV2Model,
)
from transformers.models.deit.modeling_deit import DeiTForImageClassification, DeiTForImageClassificationWithTeacher, DeiTModel
from transformers.models.detr.modeling_detr import DetrForObjectDetection, DetrModel
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForMaskedLM,
    DistilBertForMultipleChoice,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
)
from transformers.models.dpr.modeling_dpr import DPRQuestionEncoder
from transformers.models.electra.modeling_electra import (
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraModel,
)
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.models.flaubert.modeling_flaubert import (
    FlaubertForMultipleChoice,
    FlaubertForQuestionAnsweringSimple,
    FlaubertForSequenceClassification,
    FlaubertForTokenClassification,
    FlaubertModel,
    FlaubertWithLMHeadModel,
)
from transformers.models.fsmt.modeling_fsmt import FSMTForConditionalGeneration, FSMTModel
from transformers.models.funnel.modeling_funnel import (
    FunnelBaseModel,
    FunnelForMaskedLM,
    FunnelForMultipleChoice,
    FunnelForPreTraining,
    FunnelForQuestionAnswering,
    FunnelForSequenceClassification,
    FunnelForTokenClassification,
    FunnelModel,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM, GPTNeoForSequenceClassification, GPTNeoModel
from transformers.models.hubert.modeling_hubert import HubertModel
from transformers.models.ibert.modeling_ibert import (
    IBertForMaskedLM,
    IBertForMultipleChoice,
    IBertForQuestionAnswering,
    IBertForSequenceClassification,
    IBertForTokenClassification,
    IBertModel,
)
from transformers.models.layoutlm.modeling_layoutlm import (
    LayoutLMForMaskedLM,
    LayoutLMForSequenceClassification,
    LayoutLMForTokenClassification,
    LayoutLMModel,
)
from transformers.models.led.modeling_led import (
    LEDForConditionalGeneration,
    LEDForQuestionAnswering,
    LEDForSequenceClassification,
    LEDModel,
)
from transformers.models.longformer.modeling_longformer import (
    LongformerForMaskedLM,
    LongformerForMultipleChoice,
    LongformerForQuestionAnswering,
    LongformerForSequenceClassification,
    LongformerForTokenClassification,
    LongformerModel,
)
from transformers.models.luke.modeling_luke import LukeModel
from transformers.models.lxmert.modeling_lxmert import LxmertForPreTraining, LxmertForQuestionAnswering, LxmertModel
from transformers.models.m2m_100.modeling_m2m_100 import M2M100ForConditionalGeneration, M2M100Model
from transformers.models.marian.modeling_marian import MarianForCausalLM, MarianModel, MarianMTModel
from transformers.models.mbart.modeling_mbart import (
    MBartForCausalLM,
    MBartForConditionalGeneration,
    MBartForQuestionAnswering,
    MBartForSequenceClassification,
    MBartModel,
)
from transformers.models.megatron_bert.modeling_megatron_bert import (
    MegatronBertForCausalLM,
    MegatronBertForMaskedLM,
    MegatronBertForMultipleChoice,
    MegatronBertForNextSentencePrediction,
    MegatronBertForPreTraining,
    MegatronBertForQuestionAnswering,
    MegatronBertForSequenceClassification,
    MegatronBertForTokenClassification,
    MegatronBertModel,
)
from transformers.models.mobilebert.modeling_mobilebert import (
    MobileBertForMaskedLM,
    MobileBertForMultipleChoice,
    MobileBertForNextSentencePrediction,
    MobileBertForPreTraining,
    MobileBertForQuestionAnswering,
    MobileBertForSequenceClassification,
    MobileBertForTokenClassification,
    MobileBertModel,
)
from transformers.models.mpnet.modeling_mpnet import (
    MPNetForMaskedLM,
    MPNetForMultipleChoice,
    MPNetForQuestionAnswering,
    MPNetForSequenceClassification,
    MPNetForTokenClassification,
    MPNetModel,
)
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration, MT5Model
from transformers.models.openai.modeling_openai import OpenAIGPTForSequenceClassification, OpenAIGPTLMHeadModel, OpenAIGPTModel
from transformers.models.pegasus.modeling_pegasus import PegasusForCausalLM, PegasusForConditionalGeneration, PegasusModel
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetForCausalLM, ProphetNetForConditionalGeneration, ProphetNetModel
from transformers.models.rag.modeling_rag import (  # noqa: F401 - need to import all RagModels to be in globals() function
    RagModel,
    RagSequenceForGeneration,
    RagTokenForGeneration,
)
from transformers.models.reformer.modeling_reformer import (
    ReformerForMaskedLM,
    ReformerForQuestionAnswering,
    ReformerForSequenceClassification,
    ReformerModel,
    ReformerModelWithLMHead,
)
from transformers.models.retribert.modeling_retribert import RetriBertModel
from transformers.models.roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from transformers.models.roformer.modeling_roformer import (
    RoFormerForCausalLM,
    RoFormerForMaskedLM,
    RoFormerForMultipleChoice,
    RoFormerForQuestionAnswering,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerModel,
)
from transformers.models.speech_to_text.modeling_speech_to_text import Speech2TextForConditionalGeneration, Speech2TextModel
from transformers.models.squeezebert.modeling_squeezebert import (
    SqueezeBertForMaskedLM,
    SqueezeBertForMultipleChoice,
    SqueezeBertForQuestionAnswering,
    SqueezeBertForSequenceClassification,
    SqueezeBertForTokenClassification,
    SqueezeBertModel,
)
# Instead of loading the T5 from the transformers==4.9.1, we choose to load from our prefix-tuning version.
from .modeling_t5 import T5ForConditionalGeneration, T5Model
from transformers.models.tapas.modeling_tapas import (
    TapasForMaskedLM,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
)
from transformers.models.transfo_xl.modeling_transfo_xl import TransfoXLForSequenceClassification, TransfoXLLMHeadModel, TransfoXLModel
from transformers.models.visual_bert.modeling_visual_bert import VisualBertForPreTraining, VisualBertModel
from transformers.models.vit.modeling_vit import ViTForImageClassification, ViTModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForMaskedLM, Wav2Vec2ForPreTraining, Wav2Vec2Model
from transformers.models.xlm.modeling_xlm import (
    XLMForMultipleChoice,
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMModel,
    XLMWithLMHeadModel,
)
from transformers.models.xlm_prophetnet.modeling_xlm_prophetnet import (
    XLMProphetNetForCausalLM,
    XLMProphetNetForConditionalGeneration,
    XLMProphetNetModel,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaForCausalLM,
    XLMRobertaForMaskedLM,
    XLMRobertaForMultipleChoice,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaModel,
)
from transformers.models.xlnet.modeling_xlnet import (
    XLNetForMultipleChoice,
    XLNetForQuestionAnsweringSimple,
    XLNetForSequenceClassification,
    XLNetForTokenClassification,
    XLNetLMHeadModel,
    XLNetModel,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass, auto_class_update
from transformers.models.auto.configuration_auto import (
    AlbertConfig,
    BartConfig,
    BertConfig,
    BertGenerationConfig,
    BigBirdConfig,
    BigBirdPegasusConfig,
    BlenderbotConfig,
    BlenderbotSmallConfig,
    CamembertConfig,
    CanineConfig,
    CLIPConfig,
    ConvBertConfig,
    CTRLConfig,
    DebertaConfig,
    DebertaV2Config,
    DeiTConfig,
    DetrConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    EncoderDecoderConfig,
    FlaubertConfig,
    FSMTConfig,
    FunnelConfig,
    GPT2Config,
    GPTNeoConfig,
    HubertConfig,
    IBertConfig,
    LayoutLMConfig,
    LEDConfig,
    LongformerConfig,
    LukeConfig,
    LxmertConfig,
    M2M100Config,
    MarianConfig,
    MBartConfig,
    MegatronBertConfig,
    MobileBertConfig,
    MPNetConfig,
    MT5Config,
    OpenAIGPTConfig,
    PegasusConfig,
    ProphetNetConfig,
    ReformerConfig,
    RetriBertConfig,
    RobertaConfig,
    RoFormerConfig,
    Speech2TextConfig,
    SqueezeBertConfig,
    T5Config,
    TapasConfig,
    TransfoXLConfig,
    VisualBertConfig,
    ViTConfig,
    Wav2Vec2Config,
    XLMConfig,
    XLMProphetNetConfig,
    XLMRobertaConfig,
    XLNetConfig,
)


logger = logging.get_logger(__name__)


MODEL_MAPPING = OrderedDict(
    [
        # Base model mapping
        (VisualBertConfig, VisualBertModel),
        (CanineConfig, CanineModel),
        (RoFormerConfig, RoFormerModel),
        (CLIPConfig, CLIPModel),
        (BigBirdPegasusConfig, BigBirdPegasusModel),
        (DeiTConfig, DeiTModel),
        (LukeConfig, LukeModel),
        (DetrConfig, DetrModel),
        (GPTNeoConfig, GPTNeoModel),
        (BigBirdConfig, BigBirdModel),
        (Speech2TextConfig, Speech2TextModel),
        (ViTConfig, ViTModel),
        (Wav2Vec2Config, Wav2Vec2Model),
        (HubertConfig, HubertModel),
        (M2M100Config, M2M100Model),
        (ConvBertConfig, ConvBertModel),
        (LEDConfig, LEDModel),
        (BlenderbotSmallConfig, BlenderbotSmallModel),
        (RetriBertConfig, RetriBertModel),
        (MT5Config, MT5Model),
        (T5Config, T5Model),
        (PegasusConfig, PegasusModel),
        (MarianConfig, MarianMTModel),
        (MBartConfig, MBartModel),
        (BlenderbotConfig, BlenderbotModel),
        (DistilBertConfig, DistilBertModel),
        (AlbertConfig, AlbertModel),
        (CamembertConfig, CamembertModel),
        (XLMRobertaConfig, XLMRobertaModel),
        (BartConfig, BartModel),
        (LongformerConfig, LongformerModel),
        (RobertaConfig, RobertaModel),
        (LayoutLMConfig, LayoutLMModel),
        (SqueezeBertConfig, SqueezeBertModel),
        (BertConfig, BertModel),
        (OpenAIGPTConfig, OpenAIGPTModel),
        (GPT2Config, GPT2Model),
        (MegatronBertConfig, MegatronBertModel),
        (MobileBertConfig, MobileBertModel),
        (TransfoXLConfig, TransfoXLModel),
        (XLNetConfig, XLNetModel),
        (FlaubertConfig, FlaubertModel),
        (FSMTConfig, FSMTModel),
        (XLMConfig, XLMModel),
        (CTRLConfig, CTRLModel),
        (ElectraConfig, ElectraModel),
        (ReformerConfig, ReformerModel),
        (FunnelConfig, (FunnelModel, FunnelBaseModel)),
        (LxmertConfig, LxmertModel),
        (BertGenerationConfig, BertGenerationEncoder),
        (DebertaConfig, DebertaModel),
        (DebertaV2Config, DebertaV2Model),
        (DPRConfig, DPRQuestionEncoder),
        (XLMProphetNetConfig, XLMProphetNetModel),
        (ProphetNetConfig, ProphetNetModel),
        (MPNetConfig, MPNetModel),
        (TapasConfig, TapasModel),
        (MarianConfig, MarianModel),
        (IBertConfig, IBertModel),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        # Model for pre-training mapping
        (VisualBertConfig, VisualBertForPreTraining),
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (RetriBertConfig, RetriBertModel),
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForPreTraining),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (FSMTConfig, FSMTForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForPreTraining),
        (BigBirdConfig, BigBirdForPreTraining),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (MegatronBertConfig, MegatronBertForPreTraining),
        (MobileBertConfig, MobileBertForPreTraining),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForPreTraining),
        (LxmertConfig, LxmertForPreTraining),
        (FunnelConfig, FunnelForPreTraining),
        (MPNetConfig, MPNetForMaskedLM),
        (TapasConfig, TapasForMaskedLM),
        (IBertConfig, IBertForMaskedLM),
        (DebertaConfig, DebertaForMaskedLM),
        (DebertaV2Config, DebertaV2ForMaskedLM),
        (Wav2Vec2Config, Wav2Vec2ForPreTraining),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        # Model with LM heads mapping
        (RoFormerConfig, RoFormerForMaskedLM),
        (BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration),
        (GPTNeoConfig, GPTNeoForCausalLM),
        (BigBirdConfig, BigBirdForMaskedLM),
        (Speech2TextConfig, Speech2TextForConditionalGeneration),
        (Wav2Vec2Config, Wav2Vec2ForMaskedLM),
        (M2M100Config, M2M100ForConditionalGeneration),
        (ConvBertConfig, ConvBertForMaskedLM),
        (LEDConfig, LEDForConditionalGeneration),
        (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration),
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (MarianConfig, MarianMTModel),
        (FSMTConfig, FSMTForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (MegatronBertConfig, MegatronBertForMaskedLM),
        (MobileBertConfig, MobileBertForMaskedLM),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (ReformerConfig, ReformerModelWithLMHead),
        (FunnelConfig, FunnelForMaskedLM),
        (MPNetConfig, MPNetForMaskedLM),
        (TapasConfig, TapasForMaskedLM),
        (DebertaConfig, DebertaForMaskedLM),
        (DebertaV2Config, DebertaV2ForMaskedLM),
        (IBertConfig, IBertForMaskedLM),
        (MegatronBertConfig, MegatronBertForCausalLM),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Causal LM mapping
        (RoFormerConfig, RoFormerForCausalLM),
        (BigBirdPegasusConfig, BigBirdPegasusForCausalLM),
        (GPTNeoConfig, GPTNeoForCausalLM),
        (BigBirdConfig, BigBirdForCausalLM),
        (CamembertConfig, CamembertForCausalLM),
        (XLMRobertaConfig, XLMRobertaForCausalLM),
        (RobertaConfig, RobertaForCausalLM),
        (BertConfig, BertLMHeadModel),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (
            XLMConfig,
            XLMWithLMHeadModel,
        ),  # XLM can be MLM and CLM => model should be split similar to BERT; leave here for now
        (CTRLConfig, CTRLLMHeadModel),
        (ReformerConfig, ReformerModelWithLMHead),
        (BertGenerationConfig, BertGenerationDecoder),
        (XLMProphetNetConfig, XLMProphetNetForCausalLM),
        (ProphetNetConfig, ProphetNetForCausalLM),
        (BartConfig, BartForCausalLM),
        (MBartConfig, MBartForCausalLM),
        (PegasusConfig, PegasusForCausalLM),
        (MarianConfig, MarianForCausalLM),
        (BlenderbotConfig, BlenderbotForCausalLM),
        (BlenderbotSmallConfig, BlenderbotSmallForCausalLM),
        (MegatronBertConfig, MegatronBertForCausalLM),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Image Classification mapping
        (ViTConfig, ViTForImageClassification),
        (DeiTConfig, (DeiTForImageClassification, DeiTForImageClassificationWithTeacher)),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        # Model for Masked LM mapping
        (RoFormerConfig, RoFormerForMaskedLM),
        (BigBirdConfig, BigBirdForMaskedLM),
        (Wav2Vec2Config, Wav2Vec2ForMaskedLM),
        (ConvBertConfig, ConvBertForMaskedLM),
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (MBartConfig, MBartForConditionalGeneration),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (MegatronBertConfig, MegatronBertForMaskedLM),
        (MobileBertConfig, MobileBertForMaskedLM),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (ReformerConfig, ReformerForMaskedLM),
        (FunnelConfig, FunnelForMaskedLM),
        (MPNetConfig, MPNetForMaskedLM),
        (TapasConfig, TapasForMaskedLM),
        (DebertaConfig, DebertaForMaskedLM),
        (DebertaV2Config, DebertaV2ForMaskedLM),
        (IBertConfig, IBertForMaskedLM),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING = OrderedDict(
    [
        # Model for Object Detection mapping
        (DetrConfig, DetrForObjectDetection),
    ]
)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        (BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration),
        (M2M100Config, M2M100ForConditionalGeneration),
        (LEDConfig, LEDForConditionalGeneration),
        (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration),
        (MT5Config, MT5ForConditionalGeneration),
        (T5Config, T5ForConditionalGeneration),
        (PegasusConfig, PegasusForConditionalGeneration),
        (MarianConfig, MarianMTModel),
        (MBartConfig, MBartForConditionalGeneration),
        (BlenderbotConfig, BlenderbotForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (FSMTConfig, FSMTForConditionalGeneration),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (XLMProphetNetConfig, XLMProphetNetForConditionalGeneration),
        (ProphetNetConfig, ProphetNetForConditionalGeneration),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Sequence Classification mapping
        (CanineConfig, CanineForSequenceClassification),
        (RoFormerConfig, RoFormerForSequenceClassification),
        (BigBirdPegasusConfig, BigBirdPegasusForSequenceClassification),
        (BigBirdConfig, BigBirdForSequenceClassification),
        (ConvBertConfig, ConvBertForSequenceClassification),
        (LEDConfig, LEDForSequenceClassification),
        (DistilBertConfig, DistilBertForSequenceClassification),
        (AlbertConfig, AlbertForSequenceClassification),
        (CamembertConfig, CamembertForSequenceClassification),
        (XLMRobertaConfig, XLMRobertaForSequenceClassification),
        (MBartConfig, MBartForSequenceClassification),
        (BartConfig, BartForSequenceClassification),
        (LongformerConfig, LongformerForSequenceClassification),
        (RobertaConfig, RobertaForSequenceClassification),
        (SqueezeBertConfig, SqueezeBertForSequenceClassification),
        (LayoutLMConfig, LayoutLMForSequenceClassification),
        (BertConfig, BertForSequenceClassification),
        (XLNetConfig, XLNetForSequenceClassification),
        (MegatronBertConfig, MegatronBertForSequenceClassification),
        (MobileBertConfig, MobileBertForSequenceClassification),
        (FlaubertConfig, FlaubertForSequenceClassification),
        (XLMConfig, XLMForSequenceClassification),
        (ElectraConfig, ElectraForSequenceClassification),
        (FunnelConfig, FunnelForSequenceClassification),
        (DebertaConfig, DebertaForSequenceClassification),
        (DebertaV2Config, DebertaV2ForSequenceClassification),
        (GPT2Config, GPT2ForSequenceClassification),
        (GPTNeoConfig, GPTNeoForSequenceClassification),
        (OpenAIGPTConfig, OpenAIGPTForSequenceClassification),
        (ReformerConfig, ReformerForSequenceClassification),
        (CTRLConfig, CTRLForSequenceClassification),
        (TransfoXLConfig, TransfoXLForSequenceClassification),
        (MPNetConfig, MPNetForSequenceClassification),
        (TapasConfig, TapasForSequenceClassification),
        (IBertConfig, IBertForSequenceClassification),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Question Answering mapping
        (CanineConfig, CanineForQuestionAnswering),
        (RoFormerConfig, RoFormerForQuestionAnswering),
        (BigBirdPegasusConfig, BigBirdPegasusForQuestionAnswering),
        (BigBirdConfig, BigBirdForQuestionAnswering),
        (ConvBertConfig, ConvBertForQuestionAnswering),
        (LEDConfig, LEDForQuestionAnswering),
        (DistilBertConfig, DistilBertForQuestionAnswering),
        (AlbertConfig, AlbertForQuestionAnswering),
        (CamembertConfig, CamembertForQuestionAnswering),
        (BartConfig, BartForQuestionAnswering),
        (MBartConfig, MBartForQuestionAnswering),
        (LongformerConfig, LongformerForQuestionAnswering),
        (XLMRobertaConfig, XLMRobertaForQuestionAnswering),
        (RobertaConfig, RobertaForQuestionAnswering),
        (SqueezeBertConfig, SqueezeBertForQuestionAnswering),
        (BertConfig, BertForQuestionAnswering),
        (XLNetConfig, XLNetForQuestionAnsweringSimple),
        (FlaubertConfig, FlaubertForQuestionAnsweringSimple),
        (MegatronBertConfig, MegatronBertForQuestionAnswering),
        (MobileBertConfig, MobileBertForQuestionAnswering),
        (XLMConfig, XLMForQuestionAnsweringSimple),
        (ElectraConfig, ElectraForQuestionAnswering),
        (ReformerConfig, ReformerForQuestionAnswering),
        (FunnelConfig, FunnelForQuestionAnswering),
        (LxmertConfig, LxmertForQuestionAnswering),
        (MPNetConfig, MPNetForQuestionAnswering),
        (DebertaConfig, DebertaForQuestionAnswering),
        (DebertaV2Config, DebertaV2ForQuestionAnswering),
        (IBertConfig, IBertForQuestionAnswering),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Table Question Answering mapping
        (TapasConfig, TapasForQuestionAnswering),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Token Classification mapping
        (CanineConfig, CanineForTokenClassification),
        (RoFormerConfig, RoFormerForTokenClassification),
        (BigBirdConfig, BigBirdForTokenClassification),
        (ConvBertConfig, ConvBertForTokenClassification),
        (LayoutLMConfig, LayoutLMForTokenClassification),
        (DistilBertConfig, DistilBertForTokenClassification),
        (CamembertConfig, CamembertForTokenClassification),
        (FlaubertConfig, FlaubertForTokenClassification),
        (XLMConfig, XLMForTokenClassification),
        (XLMRobertaConfig, XLMRobertaForTokenClassification),
        (LongformerConfig, LongformerForTokenClassification),
        (RobertaConfig, RobertaForTokenClassification),
        (SqueezeBertConfig, SqueezeBertForTokenClassification),
        (BertConfig, BertForTokenClassification),
        (MegatronBertConfig, MegatronBertForTokenClassification),
        (MobileBertConfig, MobileBertForTokenClassification),
        (XLNetConfig, XLNetForTokenClassification),
        (AlbertConfig, AlbertForTokenClassification),
        (ElectraConfig, ElectraForTokenClassification),
        (FlaubertConfig, FlaubertForTokenClassification),
        (FunnelConfig, FunnelForTokenClassification),
        (MPNetConfig, MPNetForTokenClassification),
        (DebertaConfig, DebertaForTokenClassification),
        (DebertaV2Config, DebertaV2ForTokenClassification),
        (IBertConfig, IBertForTokenClassification),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        # Model for Multiple Choice mapping
        (CanineConfig, CanineForMultipleChoice),
        (RoFormerConfig, RoFormerForMultipleChoice),
        (BigBirdConfig, BigBirdForMultipleChoice),
        (ConvBertConfig, ConvBertForMultipleChoice),
        (CamembertConfig, CamembertForMultipleChoice),
        (ElectraConfig, ElectraForMultipleChoice),
        (XLMRobertaConfig, XLMRobertaForMultipleChoice),
        (LongformerConfig, LongformerForMultipleChoice),
        (RobertaConfig, RobertaForMultipleChoice),
        (SqueezeBertConfig, SqueezeBertForMultipleChoice),
        (BertConfig, BertForMultipleChoice),
        (DistilBertConfig, DistilBertForMultipleChoice),
        (MegatronBertConfig, MegatronBertForMultipleChoice),
        (MobileBertConfig, MobileBertForMultipleChoice),
        (XLNetConfig, XLNetForMultipleChoice),
        (AlbertConfig, AlbertForMultipleChoice),
        (XLMConfig, XLMForMultipleChoice),
        (FlaubertConfig, FlaubertForMultipleChoice),
        (FunnelConfig, FunnelForMultipleChoice),
        (MPNetConfig, MPNetForMultipleChoice),
        (IBertConfig, IBertForMultipleChoice),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = OrderedDict(
    [
        (BertConfig, BertForNextSentencePrediction),
        (MegatronBertConfig, MegatronBertForNextSentencePrediction),
        (MobileBertConfig, MobileBertForNextSentencePrediction),
    ]
)


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


_AutoModelWithLMHead = auto_class_update(_AutoModelWithLMHead, head_doc="language modeling")


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")


class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


AutoModelForMaskedLM = auto_class_update(AutoModelForMaskedLM, head_doc="masked language modeling")


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM, head_doc="sequence-to-sequence language modeling", checkpoint_for_example="t5-base"
)


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


AutoModelForSequenceClassification = auto_class_update(
    AutoModelForSequenceClassification, head_doc="sequence classification"
)


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")


class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


AutoModelForTableQuestionAnswering = auto_class_update(
    AutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)


class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


AutoModelForNextSentencePrediction = auto_class_update(
    AutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")


class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
