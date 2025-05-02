from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

batch_size = 32         # batch size of model inputting
n_best = 5              # n best probabilities found during post processing

max_answer_length = 30

def post_process_batch(tokenized_test, data_batch, example_to_features, model, device):
    t_tokenized_test = tokenized_test.remove_columns(["example_id", "offset_mapping"])
    t_tokenized_test.set_format("torch")

    test_dataloader = DataLoader(t_tokenized_test, batch_size=batch_size, shuffle=False)    # no need for shuffling

    start_probs = torch.empty((len(t_tokenized_test), 384), device=device)
    end_probs = torch.empty((len(t_tokenized_test), 384), device=device)

    model.eval()  # Set the model to evaluation mode
    # print("model inputing")
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            batch_start_probs = F.softmax(outputs.start_logits, dim=-1)
            batch_end_probs = F.softmax(outputs.end_logits, dim=-1)
            # batch_start_probs = outputs.start_logits
            # batch_end_probs = outputs.end_logits

            start_idx = i * batch_size
            end_idx = start_idx + batch_start_probs.shape[0]

            start_probs[start_idx:end_idx] = batch_start_probs
            end_probs[start_idx:end_idx] = batch_end_probs
    
    no_answer_probs = start_probs[:, 0] * end_probs[:, 0]

    start_indexes = torch.argsort(start_probs, dim=-1, descending=True)[:, :n_best] # get n best scores for each example
    end_indexes = torch.argsort(end_probs, dim=-1, descending=True)[:, :n_best]

    best_start_probs = torch.gather(start_probs, dim=1, index=start_indexes)
    best_end_probs = torch.gather(end_probs, dim=1, index=end_indexes)

    best_probs = best_start_probs.unsqueeze(2) * best_end_probs.unsqueeze(1)    # calcalate the n_best squared best probabilities for each feature
    # best_probs = best_start_probs + best_end_probs

    max_prob_indices = torch.argsort(best_probs.view((len(t_tokenized_test), -1)), dim=1, descending=True)

    # print("extracting answers")
    predicted_answers = []
    for example in data_batch:
        example_id = example["id"]
        context = example["context"]
        answers = []    # track all answers for this id

        # go through each feature that is associated to that example
        for feature_index in example_to_features[example_id]:
            offsets = tokenized_test["offset_mapping"][feature_index]   # get offset mapping
            probability_indicies = max_prob_indices[feature_index]      # get n_best squared indexes

            no_answer_probability = no_answer_probs[feature_index].cpu().item()

            # iterate through the indicies, going from maximum probability first
            for flat_idx in probability_indicies:
                start_index = start_indexes[feature_index, flat_idx // n_best]
                end_index = end_indexes[feature_index, flat_idx % n_best]

                # if the highest probability index is not part of context then move to next highest probability
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if (end_index < start_index or end_index - start_index + 1> max_answer_length):
                    continue
                # only append one set of probabilities per feature, where the prob score is the highest
                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "prob_score": best_probs[feature_index, flat_idx // n_best, flat_idx % n_best].cpu().item(),
                        "no_answer_probability": no_answer_probability,
                    }
                )
                break

        if len(answers) == 0:
            predicted_answers.append({"id": example_id, "prediction_text": "", "no_answer_probability": 1.0})
        else:
            # check for the best answer for each id
            best_answer = max(answers, key=lambda x: x["prob_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"], "no_answer_probability": best_answer["no_answer_probability"]})
        
    return predicted_answers
