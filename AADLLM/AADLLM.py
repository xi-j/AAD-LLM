# IMPORTANT:
# we have removed the check input_features.shape[-1] != expected_seq_length in Qwen2AudioEncoder
# and also modify this line: hidden_states = inputs_embeds + embed_pos[0:inputs_embeds.shape[1]]
# both in transformers/models/qwen2_audio/modeling_qwen2_audio.py
# otherwise, all audio frames are padded to 3k while attention masks are not actually used
# -> exceed GPU memory 

import torch
from torch import nn

from typing import Optional, Tuple, Union, List, Dict

from transformers import GenerationMixin
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioCausalLMOutputWithPast

import wandb

input_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAttention: <|extra_124|>\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"

class AADLLM(nn.Module, GenerationMixin):
    
    def __init__(self, 
        qwen, 
        spk_hint=False,
        spks_vects=False,
        same_proj_hint_vects=True,
        hint_tok_id=154932, # corresponding to <|extra_124|>
        spks_tok_id=154931, # corresponding to <|extra_123|>
        spk_dim=512,
        text_emb_dim=4096,
    ):
        super().__init__()
        self.DEBUG_COUNTER = 0
        
        self.qwen = qwen # LoRA-ed Qwen2AudioForConditionalGeneration
        self.config = qwen.config
        self.main_input_name = qwen.main_input_name
        self.device = qwen.device
        self.generation_config = qwen.generation_config
        self._supports_cache_class = qwen._supports_cache_class

        self.up_factor = text_emb_dim // spk_dim

        ################################################################################
        if spk_hint: # Project spk cluster to the same dimension of the text
            self.hinter = nn.Linear(
                spk_dim, text_emb_dim
            )
            print('Use a linear layer to project attended cluster onto the LLM input space.')

        if spks_vects:
            if same_proj_hint_vects:
                print('Use the same linear layer to project speaker vectors.')
            else:
                self.spk_proj = nn.Linear(
                    spk_dim, text_emb_dim
                )
                print('Use another linear layer to project speaker vectors.')
        else:
            print('Don\'t append speaker vectors.')

        self.spk_hint = spk_hint
        self.spks_vects = spks_vects
        self.same_proj_hint_vects = same_proj_hint_vects
        self.hint_tok_id = hint_tok_id
        self.spks_tok_id = spks_tok_id

        print('No additional layer between every LLM layer.')

        ################################################################################

    def can_generate(self):
        return True

    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels,
    ):
        """
        Merge input_ids with audio features into final embeddings.

        Supports two input conventions:
        1) One special audio token per audio clip
        2) Already-expanded special audio tokens, one per projected audio token
        """
        dtype = audio_features.dtype
        batch_size, n_audio, max_audio_tokens, embed_dim = audio_features.shape
        inputs_embeds = inputs_embeds.to(dtype)
        audio_features = audio_features.to(dtype)

        assert n_audio == 2, f"Expect two input audios but got {n_audio}"

        audio_features_mask = (
            torch.arange(max_audio_tokens, device=num_audio_tokens.device)
            .view(1, 1, max_audio_tokens)
            .expand(batch_size, n_audio, max_audio_tokens)
            < num_audio_tokens.unsqueeze(-1)
        )
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                left_padding = self.padding_side == "left"
            else:
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        assert left_padding, "left_padding != True"

        # 1. Find audio placeholders
        audio_token_id = self.qwen.config.audio_token_index
        special_audio_token_mask = input_ids == audio_token_id
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)  # (B,)

        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)

        batch_indices, non_audio_indices = torch.where(
            (input_ids != audio_token_id) & (attention_mask == 1)
        )

        # 2. Compute new token positions
        token_placeholder_num = torch.zeros_like(input_ids)

        # Per-sample counts
        placeholders_per_sample = special_audio_token_mask.sum(dim=-1)   # [B]
        clips_per_sample = torch.full_like(placeholders_per_sample, n_audio)  # [B], usually all 2
        total_audio_tokens_per_sample = num_audio_tokens.sum(dim=-1)     # [B]

        # Case A: one placeholder per audio clip
        case_a = placeholders_per_sample.eq(clips_per_sample)

        # Case B: already expanded placeholders, one placeholder per projected audio token
        case_b = placeholders_per_sample.eq(total_audio_tokens_per_sample)

        if not torch.all(case_a | case_b):
            debug_msg = (
                f"Audio placeholder mismatch. "
                f"placeholders_per_sample={placeholders_per_sample.tolist()}, "
                f"clips_per_sample={clips_per_sample.tolist()}, "
                f"total_audio_tokens_per_sample={total_audio_tokens_per_sample.tolist()}"
            )
            raise RuntimeError(debug_msg)

        # Fill token_placeholder_num sample by sample
        for b in range(batch_size):
            mask_b = special_audio_token_mask[b]
            n_placeholders = int(placeholders_per_sample[b].item())

            if case_a[b]:
                # one sentinel per clip -> expand each sentinel to that clip's length
                token_placeholder_num[b, mask_b] = num_audio_tokens[b].long() - 1

            elif case_b[b]:
                # already expanded -> each placeholder already corresponds to one audio token
                # so no additional expansion is needed
                token_placeholder_num[b, mask_b] = 0

            else:
                raise RuntimeError(
                    f"Unexpected sample {b}: placeholders={n_placeholders}, "
                    f"n_audio={int(clips_per_sample[b].item())}, "
                    f"total_audio_tokens={int(total_audio_tokens_per_sample[b].item())}"
                )

        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]

        if left_padding:
            new_token_positions += nb_audio_pad[:, None]

        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Allocate outputs
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num),
            self.qwen.pad_token_id,
            dtype=input_ids.dtype,
            device=inputs_embeds.device,
        )

        # 4. Place text tokens
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]

        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(
                final_attention_mask.long(), self.qwen.config.ignore_index
            ).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Place audio tokens
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False

        seq_indices = torch.arange(max_token_num, device=target_device).unsqueeze(0).expand(batch_size, max_token_num)

        if left_padding:
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            val = seq_indices < (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. "
                f"num_special_audio_tokens={num_special_audio_tokens.tolist()}, "
                f"num_audio_tokens={num_audio_tokens.tolist()}, "
                f"sum(audio_to_overwrite)={int(audio_to_overwrite.sum().item())}, "
                f"sum(num_audio_tokens)={int(num_audio_tokens.sum().item())}"
            )

        final_embedding[audio_to_overwrite] = masked_audio_features.reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(final_attention_mask == 0, 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # New for AAD-LLM
        attn_cluster: Optional[torch.FloatTensor] = None,
        spk1_vect: Optional[torch.FloatTensor] = None,
        spk2_vect: Optional[torch.FloatTensor] = None,
        task: Optional[str] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        '''
        We wrap Qwen2Audio's forward() in AADLLM's forward()
        1. Change self -> self.qwen
        2. Embed speakers in inputs_embeds
        3. Many other changes for compatibility 
        '''

        ################################################################################
        if self.spk_hint:
            assert attn_cluster != None
        if self.spks_vects:
            assert spk1_vect != None and spk2_vect != None

        assert input_ids !=  None and inputs_embeds == None
        ################################################################################

        output_attentions = output_attentions if output_attentions is not None else self.qwen.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.qwen.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.qwen.config.use_return_dict
        
        target_device = self.qwen.audio_tower.device

        if input_features is not None:
            input_features = input_features.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)

        ################################################################################
        # This section has been modified to accomodate multiple (2) audio inputs per sample

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.qwen.get_input_embeddings()(input_ids)

            # 2. Merge text and audios
            # input_features: (B, 2, 128, L) instead of (B, 128, L)
            # feature_attention_mask: (B, 2, L) instead of (B, L)

            if input_features is not None and input_ids.shape[1] != 1:
                audio_feat_lengths, audio_output_lengths = self.qwen.audio_tower._get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1)
                ) # shape (B, 2)
                # Note: audio_feat_lengths = 2 * audio_output_lengths is the REAL audio length before padding!!!
                # audio_attention_mask will contain -inf for the padded region

                batch_size, n_audio, n_mel, max_mel_seq_len = input_features.shape
                assert n_audio == 2, f'Expect two input audios but got {str(n_audio)}'
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1

                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, n_audio, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(-1).expand(batch_size, n_audio, max_seq_len)
                
                # Create mask
                padding_mask = seq_range >= lengths_expand # (batch_size, n_audio, max_seq_len)

                audio_attention_mask_ = padding_mask.view(batch_size, n_audio, 1, max_seq_len).expand(
                    batch_size, n_audio, max_seq_len, max_seq_len
                )
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.qwen.audio_tower.conv1.weight.dtype, device=self.qwen.audio_tower.conv1.weight.device
                )
                audio_attention_mask[audio_attention_mask_] = float("-inf")

                # Encode audio
                input_features = input_features.view(batch_size*n_audio, n_mel, max_mel_seq_len)
                audio_attention_mask = audio_attention_mask.view(batch_size*n_audio, audio_attention_mask.shape[-2], audio_attention_mask.shape[-1])

                # audio_attention_mask is not actually used inside the audio encoder
                audio_outputs = self.qwen.audio_tower(input_features) #, attention_mask=audio_attention_mask
                selected_audio_feature = audio_outputs.last_hidden_state

                # Project whisper3 dim to Qwen2 dim
                audio_features = self.qwen.multi_modal_projector(selected_audio_feature)
                audio_features = audio_features.view(batch_size, n_audio, *audio_features.shape[1:]) # (B, 2, L, 4096)

                # Embed audio with our modified function!!!
                inputs_embeds, attention_mask, labels, position_ids, merged_input_ids = self._merge_input_ids_with_audio_features(
                    audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
                )
                input_ids = merged_input_ids

                # torch.save(input_features.cpu(), 'debug/input_features.pt')
                # torch.save(audio_features.cpu(), 'debug/audio_features.pt')
                # torch.save(inputs_embeds.cpu(), 'debug/inputs_embeds.pt')

        ################################################################################

        # Embed attention
        if self.spk_hint:
            if not isinstance(attn_cluster, torch.Tensor):
                attn_cluster = torch.stack(attn_cluster, 0)

            hinter_weight = self.hinter.weight
            attn_cluster = attn_cluster.to(
                device=hinter_weight.device,
                dtype=hinter_weight.dtype,
            )
            attn_embeds = self.hinter(attn_cluster)  # (B, D)

            # Find the location of attention placeholders in the MERGED sequence
            attn_locs = [
                (input_ids[b] == self.hint_tok_id).nonzero(as_tuple=False).flatten()
                for b in range(len(attn_embeds))
            ]

            for b, (attn_loc, attn_embed) in enumerate(zip(attn_locs, attn_embeds)):
                attn_here = False
                if attn_loc.numel() > 0:
                    # use the first matching placeholder
                    inputs_embeds[b, attn_loc[0]] = attn_embed
                    attn_here = True

                if past_key_values is None and not attn_here:
                    print("WARNING: THE attn_embed IS NOT PUT IN!")
                    print("input_ids shape:", input_ids.shape)
                    print("num hint tokens:", (input_ids[b] == self.hint_tok_id).sum().item())

        # Embed speaker vectors
        if self.spks_vects:
            if not isinstance(spk1_vect, torch.Tensor):
                spk1_vect = torch.stack(spk1_vect, 0)
            if not isinstance(spk2_vect, torch.Tensor):
                spk2_vect = torch.stack(spk2_vect, 0)
            spk1_vect = spk1_vect.to(self.device)
            spk2_vect = spk2_vect.to(self.device)
            if self.same_proj_hint_vects:
                spk1_embeds = self.hinter(spk1_vect) # (B, D)
                spk2_embeds = self.hinter(spk2_vect) # (B, D)
                # spk1_embeds = spk1_vect.repeat_interleave(self.up_factor, dim=1)
                # spk2_embeds = spk2_vect.repeat_interleave(self.up_factor, dim=1)
            else:
                spk1_embeds = self.spk_proj(spk1_vect) # (B, D)
                spk2_embeds = self.spk_proj(spk2_vect) # (B, D)

            # Find the location of speaker placeholders
            # There should be two locations for every sample in the batch
            # corresponding to two speakers
            spks_locs = [
                (input_ids[b]==self.spks_tok_id).nonzero() for b in range(len(spk1_embeds))
            ]

            # if self.DEBUG_COUNTER < 10:
            #     torch.save(inputs_embeds.cpu(), 'debug/inputs_embeds_pre_'+str(self.DEBUG_COUNTER)+'.pt')

            # Replace the token embeddings of '<|extra_123|>'
            # by neural speaker embeddings of the same dimension
            b = 0 # Iterate over batch_size
            for spks_loc, spk1_embed, spk2_embed in zip(spks_locs, spk1_embeds, spk2_embeds):
                spk_here = False
                if spks_loc.numel():
                    spk1_loc = spks_loc[0]
                    spk2_loc = spks_loc[1]
                    inputs_embeds[b, int(spk1_loc)] = spk1_embed
                    inputs_embeds[b, int(spk2_loc)] = spk2_embed
                    spk_here = True
                if not past_key_values and not spk_here:
                    print('WARNING: THE spk1_embed/spk2_embed ARE NOT PUT IN!')
                b += 1

            # if self.DEBUG_COUNTER < 10:
            #     torch.save(inputs_embeds.cpu(), 'debug/inputs_embeds_post_'+str(self.DEBUG_COUNTER)+'.pt')
            # self.DEBUG_COUNTER += 1
        
        ################################################################################

        outputs = self.qwen.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        # Copied from Qwen2Audio
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

            if task[0] == 'extract':
                pred_spk_tok = torch.argmax(shift_logits[-4])
                attn_spk_tok = shift_labels[-4]
                wandb.log(
                    data={'train_ext':float(pred_spk_tok==attn_spk_tok)},
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output


        return Qwen2AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None, # None in first run
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # truly None
        cache_position: Optional[torch.LongTensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        num_logits_to_keep: Optional[int] = None, # truly None
        **kwargs,
    ):
        model_inputs = {}

        # Text inputs
        if past_key_values: # only the last token because previous states are in cache
            input_ids = input_ids[:, -1].unsqueeze(-1)
        model_inputs['input_ids'] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Audio inputs (idk why they are in kwargs)
        model_inputs['input_features'] = kwargs.get('input_features', None)
        model_inputs['feature_attention_mask'] = kwargs.get('feature_attention_mask', None)

        # Add batch dimension
        if model_inputs['input_features'].ndim == 3:
            model_inputs['input_features'] = model_inputs['input_features'].unsqueeze(0)
        if model_inputs['feature_attention_mask'].ndim == 2:
            model_inputs['feature_attention_mask'] = model_inputs['feature_attention_mask'].unsqueeze(0)

        # Brain decoded speaker inputs
        if self.spk_hint:
            assert 'attn_cluster' in kwargs
            model_inputs['attn_cluster'] = kwargs.get('attn_cluster', None)

        if self.spks_vects:
            assert 'spk1_vect' in kwargs and 'spk2_vect' in kwargs
            model_inputs['spk1_vect'] = kwargs.get('spk1_vect', None)
            model_inputs['spk2_vect'] = kwargs.get('spk2_vect', None)

        # Others
        model_inputs['past_key_values'] = past_key_values
        model_inputs['inputs_embeds'] = inputs_embeds
        model_inputs['use_cache'] = use_cache
        model_inputs['position_ids'] = position_ids

        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Dict = {},
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        # Move additional AAD-LLM's inputs from inputs
        # to kwargs, which will be handled later by 
        # prepare_inputs_for_generation
        if self.spk_hint:
            assert 'attn_cluster' in inputs
            attn_cluster = inputs['attn_cluster'].to(self.device)
            if attn_cluster.ndim == 1:
                attn_cluster = attn_cluster.unsqueeze(0) # (1, 512)
            kwargs['attn_cluster'] = attn_cluster
            del inputs['attn_cluster']

        if self.spks_vects:
            assert 'spk1_vect' in inputs and 'spk2_vect' in inputs
            spk1_vect = inputs['spk1_vect'].to(self.device)
            spk2_vect = inputs['spk2_vect'].to(self.device)
            if spk1_vect.ndim == 1:
                spk1_vect = spk1_vect.unsqueeze(0) # (1, 512)
            if spk2_vect.ndim == 1:
                spk2_vect = spk2_vect.unsqueeze(0) # (1, 512)
            kwargs['spk1_vect'] = spk1_vect
            kwargs['spk2_vect'] = spk2_vect
            del inputs['spk1_vect']
            del inputs['spk2_vect']

        outputs = super().generate(
            **inputs,
            generation_config=self.generation_config,
            **kwargs,
        )

        if self.spk_hint:
            inputs['attn_cluster'] = kwargs['attn_cluster']
        if self.spks_vects:
            inputs['spk1_vect'] = kwargs['spk1_vect']
            inputs['spk2_vect'] = kwargs['spk2_vect']

        return outputs
