from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_max_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum length the generated tokens can have. Corresponds to the length of the input prompt + \
                `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in \
                the prompt."
        },
    )
