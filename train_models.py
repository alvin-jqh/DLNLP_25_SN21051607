import os
import torch
from datasets import load_dataset
from datasets import DatasetDict

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer

model_name="distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

stride = 128
max_length = 384

def get_dataset():
    dataset_name = "SQuAD_2.0"

    train_path = os.path.join(os.getcwd(), "Datasets", dataset_name, "train-00000-of-00001.parquet")
    val_path = os.path.join(os.getcwd(), "Datasets", dataset_name, "validation-00000-of-00001.parquet")

    dataset = load_dataset("parquet", data_files={'train': train_path, 'val': val_path})

    reduced_val_set = dataset["val"].shuffle(seed=42).select(range(5500))
    reduced_test_set = dataset["val"].shuffle(seed=42).select(range(5500, 11000))

    reduced_set = DatasetDict({"train":dataset["train"],
                                "val":reduced_val_set,
                                "test":reduced_test_set})
    
    return reduced_set

def preprocess_train_function(examples):             # this function is used to process the training set and validation set
    questions = [q.strip() for q in examples["question"]]
    # inputs contain tokenised questions and context in same list, separated by [SEP] token
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,              # max number of tokens for each input
        truncation="only_second",           # only truncates the second thing which is the context
        stride = stride,                    # controls the number of tokens that overlap
        return_overflowing_tokens=True,     # maps each input to a question in case of long contexts
        return_offsets_mapping=True,        # keeps track of which character index each token begins and ends at
        padding="max_length",               # pads each input to the max length
    )
    # [CLS] token at the beginning of each input, [SEP] to separate question and context

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")   # gives a list of which example each input corresponds to, each example may have multiple due to truncation
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    example_ids = []

    for i, offset in enumerate(offset_mapping):
        # accounting for any overlapping
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        example_ids.append(examples["id"][sample_idx])
        # accounts for questions with no answer and use CLS token
        if len(answer["answer_start"]) == 0:
            start_char = 0
            end_char = 0
        else:
            start_char = answer["answer_start"][0]                          # get the index of the starting character
            end_char = answer["answer_start"][0] + len(answer["text"][0])   # get the index of the end character

        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        # At the question, sequence id = 0, for context sequence id = 1, else it is None
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        elif start_char == 0 and end_char == 0:             
            # if there is no answer and the answer is set to CLS, label (0, 0)
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:  # while before the end of context and start idx of token <= start idx of answer
                idx += 1
            start_positions.append(idx - 1)

            idx = context_start
            while idx <= context_end and offset[idx][1] <= end_char:
                idx += 1
            end_positions.append(idx - 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["example_id"] = example_ids
    return inputs


def train(dataset, output_dir, batch_size, epochs, lr, weight_decay, device):
    print("tokenizing training data\n")
    tokenized_train = dataset["train"].map(
        preprocess_train_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print("tokenizing validation data\n")
    tokenized_eval = dataset["val"].map(
        preprocess_train_function,
        batched=True,
        remove_columns=dataset["val"].column_names,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    steps_per_epoch = len(tokenized_train) // batch_size + 1
    eval_steps = max(1, steps_per_epoch // 3)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        save_strategy="steps",  # Match this with eval
        logging_strategy="steps",
        logging_steps=eval_steps,
        eval_steps=eval_steps,
        save_steps=eval_steps,  # Add this line
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        push_to_hub=False,
        load_best_model_at_end=True,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
    )

    print("Training start\n")
    trainer.train()

    return model
