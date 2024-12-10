

import os
os.environ["WANDB_DISABLED"] = "true"
!pip install transformers datasets evaluate torch

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import time

# Load the IMDb dataset
dataset = load_dataset("imdb")

train_data = dataset["train"]
test_data = dataset["test"]

# Define a custom configuration for BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 2

model = AutoModelForSequenceClassification.from_config(config)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",v
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Training model...")
start_time = time.time()
trainer.train()

# Measure GPU memory usage after training
if torch.cuda.is_available():
    print(f"GPU Memory Usage After Training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# 7. Evaluate the model
print("Evaluating model...")
start_time = time.time()
results = trainer.evaluate()

# Measure GPU memory usage after evaluation
if torch.cuda.is_available():
    print(f"GPU Memory Usage After Evaluation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time (evaluation on the test set): {inference_time:.2f} seconds")
print(results)

accuracy = results.get("eval_accuracy", None)
precision = results.get("eval_precision", None)
recall = results.get("eval_recall", None)
f1_score = results.get("eval_f1", None)
roc_auc = results.get("eval_roc_auc", None)

# GPU memory usage during evaluation and training
gpu_memory_after_training = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else None
gpu_memory_after_evaluation = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else None

# Display the metrics
print(f"Accuracy: {accuracy:.4f}" if accuracy is not None else "Accuracy metric not found.")
print(f"Precision: {precision:.4f}" if precision is not None else "Precision metric not found.")
print(f"Recall: {recall:.4f}" if recall is not None else "Recall metric not found.")
print(f"F1-Score: {f1_score:.4f}" if f1_score is not None else "F1-Score metric not found.")
print(f"ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC metric not found.")

# Displaying GPU memory usage
if gpu_memory_after_training:
    print(f"GPU Memory Usage After Training: {gpu_memory_after_training:.2f} GB")
if gpu_memory_after_evaluation:
    print(f"GPU Memory Usage After Evaluation: {gpu_memory_after_evaluation:.2f} GB")
