import logging

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from utils import DataCollatorForGit, get_dataset

logger = logging.getLogger(__name__)

from arguments import DatasetsArguments, ModelArguments, MyTrainingArguments


def main(model_args: ModelArguments, data_args: DatasetsArguments, training_args: MyTrainingArguments):
    set_seed(training_args.seed)

    dataset = get_dataset(csv_path=data_args.train_data_path)
    dataset = dataset.train_test_split(test_size=0.01, seed=training_args.seed)

    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    data_collator = DataCollatorForGit(processor=processor)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()

    if is_main_process(training_args.local_rank):
        model.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    main(model_args=model_args, data_args=data_args, training_args=training_args)
