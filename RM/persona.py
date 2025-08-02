import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig
import logging
import os
import json

logger = logging.getLogger(__name__)

class PersonaEncoderDecoder(nn.Module):
    """
    Encodes a persona's domain knowledge into a continuous embedding
    and decodes that embedding into a classification of persona type.
    """
    def __init__(self, model_name="microsoft/deberta-v3-base", embedding_dim=256, num_classes=3, use_lora=False, lora_config=None):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.projection_dim = 512

        # --- Encoder Components ---
        self.foundation_model = AutoModel.from_pretrained(model_name)
        if use_lora:
            # Apply LoRA to the foundation model for efficient fine-tuning
            lora_config = lora_config or LoraConfig(r=16, lora_alpha=32, target_modules=['query_proj', 'value_proj'], lora_dropout=0.05, bias="none")
            self.foundation_model = get_peft_model(self.foundation_model, lora_config)
            logger.info("Enabled LoRA for the foundation model.")
        else:
            # Freeze foundation model if not using LoRA
            for param in self.foundation_model.parameters():
                param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(self.foundation_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_dim)
        )

        # --- Decoder Component ---
        self.decoder_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

        # --- Internal storage for outputs ---
        self._last_embedding = None
        self._last_classification_logits = None

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Processes the input text and returns logits.
        If labels are provided, it also returns the CrossEntropyLoss.
        """
        # --- Encoding Step ---
        foundation_outputs = self.foundation_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = foundation_outputs.last_hidden_state

        # Masked mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_embedding = sum_embeddings / sum_mask

        # Project to persona embedding
        persona_embedding = self.projection_head(pooled_embedding)
        self._last_embedding = persona_embedding  # Store for later access

        # --- Decoding Step ---
        logits = self.decoder_classifier(persona_embedding)
        self._last_classification_logits = logits # Store for later access

        # --- Loss Calculation ---
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        if not self.training:
            return (loss, logits, persona_embedding) if loss is not None else (logits, persona_embedding)

        return (loss, logits) if loss is not None else logits

    def save_pretrained(self, save_directory):
        """
        Saves the model weights and configuration to a directory, making it compatible
        with the Huggingface ecosystem.
        """
        logger.info(f"Saving model to {save_directory}")
        os.makedirs(save_directory, exist_ok=True)

        # Save the configuration
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "projection_dim": self.projection_dim,
            "use_lora": hasattr(self.foundation_model, 'adapter_name')
        }
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Save the foundation model (handles LoRA adapters correctly)
        self.foundation_model.save_pretrained(save_directory)

        # Save the custom heads
        custom_heads_state_dict = {
            'projection_head': self.projection_head.state_dict(),
            'decoder_classifier': self.decoder_classifier.state_dict()
        }
        torch.save(custom_heads_state_dict, os.path.join(save_directory, 'custom_heads.bin'))

    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Loads the model from a directory saved with `save_pretrained`.
        """
        logger.info(f"Loading model from {load_directory}")

        # Load configuration
        config_path = os.path.join(load_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Initialize the model with the saved configuration
        model = cls(
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            use_lora=config['use_lora']
        )

        # The foundation model is loaded automatically by the cls constructor.
        # If LoRA was used, the adapters are loaded with the foundation model.

        # Load the custom heads
        custom_heads_path = os.path.join(load_directory, 'custom_heads.bin')
        if os.path.exists(custom_heads_path):
            custom_heads_state_dict = torch.load(custom_heads_path, map_location=torch.device("cpu"))
            model.projection_head.load_state_dict(custom_heads_state_dict['projection_head'])
            model.decoder_classifier.load_state_dict(custom_heads_state_dict['decoder_classifier'])
        else:
            logger.warning(f"Could not find custom_heads.bin in {load_directory}. The projection and classifier heads are not loaded.")

        return model

    def get_last_embedding(self):
        """Returns the last computed persona embedding."""
        return self._last_embedding

    def get_last_classification_logits(self):
        """Returns the last computed classification logits from the decoder."""
        return self._last_classification_logits