from src.data_preprocessing import load_data
from src.dataset import create_data_loaders, SentimentDataset
from src.model import train_model
from src.evaluation import evaluate_model
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import yaml

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data, test_data = load_data(config['data_path'], config['sample_size'])

    # Setup tokenizer and data loaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader, test_loader = create_data_loaders(train_data, test_data, tokenizer, config['batch_size'])

    # Setup model and optimizer
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

    # Train model
    model = train_model(model, train_loader, optimizer, device, config['epochs'])

    # Evaluate model
    accuracy, report = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy}")
    print(f"\nClassification Report:\n{report}")

    # Save model
    save_model(model, config['model_save_path'])

if __name__ == "__main__":
    main()
