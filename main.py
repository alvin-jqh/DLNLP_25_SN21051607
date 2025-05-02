from train_models import train, get_dataset
import os
import torch

from evaluation import test
from transformers import AutoModelForQuestionAnswering

# ======================================================================================================================
# Data preprocessing
dataset = get_dataset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

choice = input("Do you want to: \n1. train for scratch \n2. evaluate from saved \nEnter 1 or 2 \n")

if choice == '1':
    print("Training from scratch...")
    output_name = input("Type the name of the file you want to save the model is in\n")
    output_dir = os.path.join(os.getcwd(), "models", output_name)

    batch_size = int(input("Enter batch size\n"))
    epochs = int(input("How many epochs do you want to train for?\n"))
    learning_rate = float(input("Enter the learning rate\n"))
    weight_decay = float(input("Enter the weight decay\n"))

    model = train(dataset, output_dir, batch_size, epochs, learning_rate, weight_decay)
    print("Evaluating")

    metrics = test(dataset, model, device)
    
    print(metrics)

elif choice == '2':
    print("Evaluating from saved model...")
    model_name = input("Type the name of the file you want to load the model from\n")
    model_dir = os.path.join(os.getcwd(), "models", model_name, "checkpoint-24705")

    trained_model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)

    metrics = test(dataset, trained_model, device)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
else:
    print("Invalid input. Please enter 1 or 2.")





# ======================================================================================================================
# # Task A
# model_A = train(dataset, output_dir, batch_size, learning_rate, weight_decay)

# acc_A_test = model_A.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.




# # ======================================================================================================================
# ## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A_train = 'TBD'
# # acc_B_test = 'TBD'