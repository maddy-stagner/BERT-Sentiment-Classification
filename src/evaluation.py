import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    return accuracy_score(real_values, predictions), classification_report(real_values, predictions)
