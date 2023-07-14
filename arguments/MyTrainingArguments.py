from dataclasses import dataclass

from transformers import Seq2SeqTrainingArguments


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    # https://github.com/huggingface/transformers/blob/v4.22.2/src/transformers/training_args_seq2seq.py#L28
    pass
