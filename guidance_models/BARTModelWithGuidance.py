import os
import random
import re
from typing import Optional, Tuple, Union, Dict, Any, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BartTokenizer, cached_path, DataCollatorForSeq2Seq, PreTrainedTokenizerBase
from transformers.file_utils import hf_bucket_url, ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import BartModel, BartEncoder, BartDecoder, BartConfig, \
    BartForConditionalGeneration, BartDecoderLayer, BartAttention, _expand_mask, shift_tokens_right
from transformers.utils import PaddingStrategy

from transformers import Seq2SeqTrainer

"""
Heavily adpated from https://github.com/huggingface/transformers/blob/v4.19.4/src/transformers/models/bart/modeling_bart.py

Couldn't make it any DRY-er as a lot of the changes are within internal of each of the classes.
I've tried to specify where in a method a change has occurred.

Guidance implementation based off of this paper: 
Y. Dou et al. "GSum: A General Framework for Guided Neural Abstractive Summarization" in Proc. NAACL (2021).
"""

class BartModelWithGuidance(BartModel):

    def __init__(self, config: BartConfig, shared_encoder_layers=3):
        r"""

        :param config: regular BART config
        :param shared_encoder_layers: default -1 means all layers up until last encoder are shared.
        """
        super(BartModelWithGuidance, self).__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.guidance_encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoderWithGuidance(config, self.shared)

        # share first X encoder layers
        if shared_encoder_layers == -1 or shared_encoder_layers >= self.config.encoder_layers:
            shared_encoder_layers_lim: int = self.config.encoder_layers - 1
        else:
            shared_encoder_layers_lim: int = shared_encoder_layers

        for i in range(shared_encoder_layers_lim):
            shared_layer = self.encoder.layers[i]
            self.guidance_encoder.layers[i] = shared_layer

        super(BartModelWithGuidance, self).init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_guidance_ids=None,
        attention_guidance_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        cross_attn_guidance_mask=None,
        encoder_outputs=None,
        encoder_guidance_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """ Mostly copied from hf modelling_bart.BartModel super class, but with additional params for
        encoder_guidance processing"""
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        ###  Additional guidance encoder  ####
        if encoder_guidance_outputs is None:
            encoder_guidance_outputs = self.guidance_encoder(
                input_ids=input_guidance_ids,
                attention_mask=attention_guidance_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            encoder_guidance_hidden_states=encoder_guidance_outputs[0],
            encoder_guidance_attention_mask=attention_guidance_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            cross_attn_guidance_mask=cross_attn_guidance_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            # missing the guidance encoder outputs
        )


class BartDecoderLayerWithGuidance(BartDecoderLayer):
    def __init__(self, config: BartConfig):
        super(BartDecoderLayerWithGuidance, self).__init__(config)
        # extra cross attention + layernorm params
        self.encoder_guidance_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_guidance_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_guidance_hidden_states: Optional[torch.Tensor] = None,
        encoder_guidance_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_guidance_layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,   # don't use cache, this is True in super.
    ):
        """ Mostly copied from hf modelling_bart.BartDecoderLayer super class, but with additional params for guidance encoder,
        hidden_states and attention_mask etc."""
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        ####  Guidance Cross Attention Block  ####
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_guidance_hidden_states is not None:
            residual = hidden_states
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_guidance_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_guidance_hidden_states,
                attention_mask=encoder_guidance_attention_mask,
                layer_head_mask=cross_attn_guidance_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_guidance_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
        ###   END - GUIDANCE Decoder Cross Attention Block  ####

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartDecoderWithGuidance(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super(BartDecoderWithGuidance, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList([BartDecoderLayerWithGuidance(config) for _ in range(config.decoder_layers)])
        # copy params ??? from DecoderLayer attn matrices ?? to utilise pre-training
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_guidance_hidden_states=None,
        encoder_guidance_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        cross_attn_guidance_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """ Mostly copied from hf modelling_bart.BartDecoder super class, but with additional params for
         guidance encoder / cross-attention computations etc."""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if encoder_guidance_hidden_states is not None and encoder_guidance_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_guidance_attention_mask = _expand_mask(encoder_guidance_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_guidance_hidden_states=encoder_guidance_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_guidance_attention_mask=encoder_guidance_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                cross_attn_guidance_layer_head_mask=(
                    cross_attn_guidance_mask[idx] if cross_attn_guidance_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartWithGuidanceForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super(BartWithGuidanceForConditionalGeneration, self).__init__(config)
        self.model = BartModelWithGuidance(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_guidance_ids=None,
        attention_guidance_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        encoder_guidance_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            input_guidance_ids=input_guidance_ids,
            attention_guidance_mask=attention_guidance_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            encoder_guidance_outputs=encoder_guidance_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "encoder_guidance_outputs": kwargs['encoder_guidance_outputs'],
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "attention_guidance_mask": kwargs['attention_guidance_mask'],
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """ largely copied over from transformers.generation_utils.GenerationMixin """
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            # encoder = self.get_encoder()
            encoder = self.model.encoder
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn")
                        or argument in ['input_guidance_ids', 'attention_guidance_mask', 'use_cache'])
            }
            encoder_outputs: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)

            ##### Prepare Guidance Encoder IDs / attention mask for encoding ######
            guidance_encoder = self.model.guidance_encoder
            input_guidance_ids = model_kwargs['input_guidance_ids']
            attention_guidance_mask = model_kwargs['attention_guidance_mask']
            guidance_encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn") or
                        argument in ['input_ids', 'input_guidance_ids', 'attention_guidance_mask', 'use_cache'])
            }
            guidance_encoder_kwargs['attention_mask'] = attention_guidance_mask
            encoder_guidance_outputs: ModelOutput = guidance_encoder(input_guidance_ids, return_dict=True,
                                                                     **guidance_encoder_kwargs)
            model_kwargs['encoder_outputs'] = encoder_outputs
            model_kwargs['encoder_guidance_outputs'] = encoder_guidance_outputs
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        input_ids, model_kwargs = super(BartWithGuidanceForConditionalGeneration,
                                        BartWithGuidanceForConditionalGeneration)\
            ._expand_inputs_for_generation(input_ids, expand_size=expand_size,
                                           is_encoder_decoder=is_encoder_decoder,
                                           attention_mask=attention_mask,
                                           encoder_outputs=encoder_outputs, **model_kwargs)
        input_guidance_ids = model_kwargs['input_guidance_ids']
        expanded_return_idx = (
            torch.arange(input_guidance_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_guidance_ids.device)
        )
        model_kwargs['input_guidance_ids'] = input_guidance_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs['attention_guidance_mask'] = \
                model_kwargs['attention_guidance_mask'].index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert model_kwargs['encoder_guidance_outputs'] is not None
            lhs = model_kwargs['encoder_guidance_outputs'].last_hidden_state
            model_kwargs['encoder_guidance_outputs']['last_hidden_state'] = \
                lhs.index_select(0, expanded_return_idx.to(lhs.device))

        return input_ids, model_kwargs


def adapted_state_dict(pretrained_model_name_or_path: str, **kwargs):
    # share params from text encoder to guidance encoder.
    try:
        # attempt to load raw path
        path = cached_path(pretrained_model_name_or_path + '/pytorch_model.bin', **kwargs)
    except OSError:
        path = cached_path(hf_bucket_url(pretrained_model_name_or_path, filename='pytorch_model.bin'), **kwargs)

    state_dict = torch.load(path, map_location="cpu")
    # copy filled params to missing encoder / decoder guidance_<foo> etc.
    guidance_encoder_params = {'model.guidance_encoder.' + '.'.join(k.split('.')[2:]): v for k, v in state_dict.items() if k.startswith('model.encoder')}
    decoder_encoder_guidance_attn_params = {k.replace('encoder_attn', 'encoder_guidance_attn'): v for k, v in state_dict.items()
                                            if re.match(r'model\.decoder\.layers\.[0-9]+\.encoder_attn\.', k)}
    decoder_encoder_guidance_attn_layer_norm_params = {k.replace('encoder_attn', 'encoder_guidance_attn'): v for k, v in state_dict.items()
                                                       if re.match(r'model\.decoder\.layers\.[0-9]+\.encoder_attn_layer_norm', k)}

    revised_state_dict = dict(state_dict, **guidance_encoder_params, **decoder_encoder_guidance_attn_params,
                              **decoder_encoder_guidance_attn_layer_norm_params)
    return revised_state_dict


def pad_guidance_signal(input_ids, input_guidance_ids, attention_guidance_mask, tokenizer):
    """ Used to pad the guidance signal / attention mask to the same size as the input,
        assumes the guidance signal is smaller than the raw input.
     """
    ### ensure consistent dim for guidance and input signal ###
    len_to_pad = input_ids.shape[-1] - input_guidance_ids.shape[-1]
    input_guidance_ids = torch.cat([input_guidance_ids, torch.full((1, len_to_pad), tokenizer.pad_token_id)], 1)
    attention_guidance_mask = torch.cat([attention_guidance_mask, torch.zeros((1, len_to_pad))], 1)
    return input_guidance_ids, attention_guidance_mask


class DataCollatorForBartGuidanceSeq2Seq(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        in_features = [{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'labels']}
                       for inputs in features]

        in_guidance_features = []
        for inputs in features:
            in_guidance_features.append({
                'input_ids': inputs['input_guidance_ids'],
                'attention_mask': inputs['attention_guidance_mask']
            })
        features = self.tokenizer.pad(
            in_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        guidance_features = self.tokenizer.pad(
            in_guidance_features,
            padding='max_length',
            max_length=features.input_ids.shape[-1],
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors
        )

        features['input_guidance_ids'] = guidance_features['input_ids']
        features['attention_guidance_mask'] = guidance_features['attention_mask']
        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


class BartGuidanceSeq2SeqTrainer(Seq2SeqTrainer):

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ Implementation is the equivalent to the super class but includes the extra inputs for the
        guidance encoder, "inputs_guidance_ids" and "attention_guidance_mask"
        There's no DRY-er way to do it unfortunately.
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        #####  include guidance signal for prediction inputs  ######
        gen_kwargs['input_guidance_ids'] = inputs['input_guidance_ids']
        gen_kwargs['attention_guidance_mask'] = inputs['attention_guidance_mask']
        ###### end extra inputs to be sent to model ######

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)


if __name__ == '__main__':
    # model = BartModelWithGuidance.from_pretrained('sshleifer/bart-tiny-random')
    # model_str = 'sshleifer/bart-tiny-random'
    model_str = 'sshleifer/distilbart-cnn-6-6'
    # model_str = 'facebook/bart-base'
    # model = BartModel.from_pretrained(model_str)
    model = BartWithGuidanceForConditionalGeneration.from_pretrained(model_str,
                                                                     state_dict=adapted_state_dict(model_str))
    # model = BartForConditionalGeneration.from_pretrained(model_str)
    tok = BartTokenizer.from_pretrained(model_str)
    example_english_phrase = "UN Chief Says There Is No Threat in Syria although there is significant instability in the area"
    batch = tok(example_english_phrase, return_tensors="pt")
    example_guidance_phrase = "UN Threat Syria"
    batch_guidance = tok(example_guidance_phrase, return_tensors="pt")
    input_guidance_ids, attention_guidance_mask = pad_guidance_signal(batch["input_ids"], batch_guidance['input_ids'],
                                                                       batch_guidance['attention_mask'], tok)
    generated_ids = model.generate(batch["input_ids"], attention_mask=batch['attention_mask'],
                                   input_guidance_ids=input_guidance_ids,
                                   attention_guidance_mask=attention_guidance_mask)
    # generated_ids = model.generate(batch["input_ids"])
    print(tok.batch_decode(generated_ids, skip_special_tokens=True))



