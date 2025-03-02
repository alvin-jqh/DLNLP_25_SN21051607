from transformers import AutoTokenizer
import collections
import numpy as np
import evaluate
from tqdm.auto import tqdm

stride = 128
max_length = 384

n_best = 5
max_answer_length = 20

model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

metric = evaluate.load("squad_v2")

def preprocess_train_function(examples):
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
    sample_map = inputs.pop("overflow_to_sample_mapping")
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
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        # At the question, sequence id = 0, for context sequence id = 1
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
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["example_id"] = example_ids
    return inputs

def preprocess_val_function(examples):
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

    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")
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
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        offset_mapping[i] = [
            (o if s == 1 else None) for o, s in zip(offset, sequence_ids)
        ]

        # Find the start and end of the context
        # At the question, sequence id = 0, for context sequence id = 1
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
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["example_id"] = example_ids
    return inputs

def post_process(start_probs, end_probs, features, examples):
    """
    Args:
        start_probs: softmaxed output.start_logits
        end_probs: softmaxed output.end_logits
        features: tokenized inputs
        examples: datapoints from the dataset
    """

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
    predicted_answers = []
    references = []

    # for each question 
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        # and each feature, if the context was truncated
        for feature_index in example_to_features[example_id]:
            start_prob = start_probs[feature_index]
            end_prob = end_probs[feature_index]
            offsets = features["offset_mapping"][feature_index]

            start_indexes = np.argsort(start_prob)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_prob)[-1 : -n_best - 1 : -1].tolist() 

            no_answer_prob = start_prob[0] * end_prob[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    answers.append(
                        {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "score": start_prob[start_index] * end_prob[end_index],
                            "no_answer_probability": no_answer_prob,
                        }
                    )

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"], "no_answer_probability":best_answer["no_answer_probability"]})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": "", "no_answer_probability":1.0})

        references.append({"id": example_id, "answers": example["answers"]})

    return metric.compute(predictions=predicted_answers, references=references), predicted_answers, references
