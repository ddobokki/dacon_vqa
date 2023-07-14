import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import datasets
import pandas as pd
from datasets import Dataset
from transformers import GitProcessor

from literal import ANSWER, IMG, IMG_PATH, QUESTION


@dataclass
class DataCollatorForGit:
    processor: GitProcessor
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [feature[IMG].convert("RGB") for feature in features]

        if ANSWER in features[0]:
            questions = [
                feature[QUESTION] + self.processor.tokenizer.sep_token + feature[ANSWER] for feature in features
            ]
        else:
            questions = [feature[QUESTION] for feature in features]
        batch = self.processor(images=images, return_tensors=self.return_tensors)
        tokenized_question = self.processor.tokenizer(
            questions, padding=self.padding, return_tensors=self.return_tensors
        )
        batch["input_ids"] = tokenized_question.input_ids
        batch["attention_mask"] = tokenized_question.attention_mask
        if ANSWER in features[0]:
            batch["labels"] = batch["input_ids"]
        return batch


def get_dataset(csv_path: os.PathLike) -> Dataset:
    df = pd.read_csv(csv_path)

    data_dict = {
        IMG: df[IMG_PATH].tolist(),
        QUESTION: df[QUESTION].tolist(),
    }

    if ANSWER in df.columns:
        data_dict[ANSWER] = df[ANSWER].tolist()

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column(IMG, datasets.Image())
    return dataset
