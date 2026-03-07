import os
import wandb
import torch

from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from peft import set_peft_model_state_dict, get_peft_model_state_dict

from typing import Optional


def save_lora(lora_model, path):
    peft_state_dict = get_peft_model_state_dict(lora_model)
    torch.save(peft_state_dict, path)

def load_lora(lora_model, path):
    peft_state_dict = torch.load(path)
    result = set_peft_model_state_dict(lora_model, peft_state_dict)


class TrainerWithAttention(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        ################################################################################
        if 'wandb' in self.args.report_to:
            assert len(inputs['task']) == 1 # per_device_batch_size == 1
            task = inputs['task'][0]
            wandb.log(
                data={task+'_loss':float(loss)},
            )
        
        ################################################################################
            
        return (loss, outputs) if return_outputs else loss

    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        ################################################################################
        
        save_lora(
            self.model.qwen, 
            os.path.join(output_dir, 'lora.pt')
        )

        if self.model.spk_hint:
            torch.save(
                self.model.hinter.state_dict(), 
                os.path.join(output_dir, 'hinter.pt')
            )

        # if self.model.spks_vects and not self.model.same_proj_hint_vects:
        #     torch.save(
        #         self.model.spk_proj.state_dict(), 
        #         os.path.join(output_dir, 'spk_proj.pt')
        #     )
        
        ################################################################################
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)   
