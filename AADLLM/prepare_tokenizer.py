from transformers import AutoTokenizer

def prepare(spk_hint=False, model='Qwen/Qwen2-Audio-7B-Instruct'):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model,
        trust_remote_code=True
    )
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<|extra_123|>']}
    )
    if spk_hint:
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<|extra_123|>', '<|extra_124|>']}
        )
    return tokenizer
