from post_process import post_process_batch
import evaluate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import collections
from tqdm import tqdm

model_name="distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

stride = 128
max_length = 384

data_batch_size = 100

def preprocess_test_function(examples):              # used to process the test set for validation
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def tokenize_batches(data_batches):
    tokenized_inputs = []
    for batch in data_batches:
        tokenized_test = batch.map(
            preprocess_test_function,
            batched=True,
            remove_columns=batch.column_names,
        )

        tokenized_inputs.append(tokenized_test)

    return tokenized_inputs

def map_examples(tokenized_inputs):
    example_mappings=[]

    for tokenized_test in tokenized_inputs:
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(tokenized_test):
            example_to_features[feature["example_id"]].append(idx)
        
        example_mappings.append(example_to_features)

    return example_mappings

def test(dataset, model, device, threshold = 0.0728035643696785):
    # seperate the dataset into smaller batches
    data_batches = [dataset["test"].select(range(i,i+data_batch_size)) for i in range(0, len(dataset["test"]), data_batch_size)]
    # tokenize the batches
    tokenized_inputs = tokenize_batches(data_batches)
    # map the dataset to their example ids
    example_mappings = map_examples(tokenized_inputs)

    predicted_answers = []

    # post process each batch, getting all the predictions
    for idx, (tokenized_test, data_batch, example_mapping) in enumerate(zip(tokenized_inputs, data_batches, example_mappings)):
        print(f"Batch {idx + 1} / {len(data_batches)}")
        pp = post_process_batch(tokenized_test, data_batch, example_mapping, model, device)
        predicted_answers = predicted_answers + pp

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset["test"]]

    metric = evaluate.load("squad_v2")
    return metric.compute(predictions=predicted_answers, references=references, no_answer_threshold = threshold)

    