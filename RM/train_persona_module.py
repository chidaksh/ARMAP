import os
import json
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer

# Make sure persona.py is in the python path
from persona import PersonaEncoderDecoder

logger = logging.getLogger(__name__)

class PersonaDataset(Dataset):
    """
    Loads persona knowledge and labels from a JSON file.
    Each item in the JSON file should be a dictionary like:
    {"knowledge_file_path": "path/to/doc.txt", "persona_label": 1}
    """
    def __init__(self, json_path, tokenizer, max_length=1024):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        knowledge_path = item['knowledge_file_path']
        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                knowledge_text = f.read()
        except FileNotFoundError:
            logger.error(f"Knowledge file not found: {knowledge_path}")
            # Return a dummy item or raise an error
            # For now, let's raise it to stop execution if a file is missing
            raise

        tokenized_inputs = self.tokenizer(
            knowledge_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Squeeze tensors to remove the batch dimension added by the tokenizer
        input_ids = tokenized_inputs['input_ids'].squeeze(0)
        attention_mask = tokenized_inputs['attention_mask'].squeeze(0)
        label = torch.tensor(item['persona_label'], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }

def run_inference(checkpoint_path, knowledge_file_path, device):
    """
    Runs inference on a single knowledge file using a trained model checkpoint.
    """
    logger.info(f"--- Running Inference ---")
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    logger.info(f"Knowledge file: {knowledge_file_path}")

    # Load the trained model and tokenizer
    model = PersonaEncoderDecoder.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    # Load and tokenize the knowledge text
    try:
        with open(knowledge_file_path, 'r', encoding='utf-8') as f:
            knowledge_text = f.read()
    except FileNotFoundError:
        logger.error(f"Inference knowledge file not found: {knowledge_file_path}")
        return

    # 3. Process text and perform inference
    inputs = tokenizer(
        knowledge_text, 
        return_tensors="pt", 
        max_length=1024, # Should match training max_length
        truncation=True, 
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        _ = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # 4. Retrieve and print outputs
    persona_embedding = model.get_last_embedding()
    classification_logits = model.get_last_classification_logits()
    predicted_class_id = torch.argmax(classification_logits, dim=-1).item()

    logger.info(f"Persona Embedding Shape: {persona_embedding.shape}")
    logger.info(f"Predicted Persona Class ID: {predicted_class_id}")
    logger.info("--- Inference Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Train the PersonaEncoderDecoder model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints.")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="Foundation model to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--use_lora", action='store_true', help="Enable LoRA for fine-tuning.")
    parser.add_argument("--inference_knowledge_file", type=str, default=None, help="Path to a knowledge file to run inference on after training.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 1. Instantiate Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PersonaEncoderDecoder(
        model_name=args.model_name,
        use_lora=args.use_lora
    )

    # 2. Load Data
    dataset = PersonaDataset(json_path=args.dataset_path, tokenizer=tokenizer)

    # 3. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none", # Can be changed to "tensorboard" or "wandb"
        fp16=torch.cuda.is_available(), # Use mixed precision if a GPU is available
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # You can add an eval_dataset here for evaluation during training
    )

    # 5. Start Training
    logger.info("Starting model training...")
    trainer.train()

    # 6. Save the final model and tokenizer
    final_checkpoint_path = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)
    logger.info(f"Final trained persona model saved to {final_checkpoint_path}")

    # 7. Run inference if a file is provided
    if args.inference_knowledge_file:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_inference(
            checkpoint_path=final_checkpoint_path,
            knowledge_file_path=args.inference_knowledge_file,
            device=device
        )

if __name__ == "__main__":
    main()
