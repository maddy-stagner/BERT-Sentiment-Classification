from transformers import BertForSequenceClassification, AdamW
import torch
from tqdm.auto import tqdm

def train_model(model, train_loader, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()
            _, predictions = torch.max(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            running_loss += loss.item()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        epoch_accuracy = correct_predictions / total_predictions
        average_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Training Loss: {average_loss:.3f}, Accuracy: {epoch_accuracy:.3f}')
    return model
