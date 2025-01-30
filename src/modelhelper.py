import ipdb
import os
import json
import math
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.utils import PretrainedModelMixin

logger = logging.getLogger(__name__)

class CodeModel(nn.Module, PretrainedModelMixin):
    """
    CodeModel: A transformer-based model for code analysis and generation.
    
    The model consists of:
    - Token and position embeddings
    - Transformer encoder layers
    - Output layer for token logits and switch predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Set default configuration values
        default_config = {
            "max_length": 10,
            "d_model": 512,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 2048,
            "dropout": 0.2,
            "alpha_distill": 0.6,  # Distillation loss weight
            "alpha_ce": 0.2,      # Cross entropy loss weight 
            "alpha_switch": 0.2,  # Switch loss weight
        }
        
        # Ensure vocab_size is provided
        if "vocab_size" not in config:
            raise ValueError("vocab_size must be provided in config")
            
        # Update defaults with provided config
        self.config = default_config.copy()
        self.config.update(config)
        
        # Create model components
        self.wte = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        self.wpe = nn.Embedding(self.config["max_length"], self.config["d_model"])
        self.drop = nn.Dropout(self.config["dropout"])
        
        # Create transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["dropout"],
            batch_first=True,
            norm_first=True  # Change this to False to enable nested tensor optimization
        )
        
        # Create transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.config["num_layers"]
        )
        
        # Output layers
        self.ln_f = nn.LayerNorm(
            self.config["d_model"], 
        )
        self.output = nn.Linear(self.config["d_model"], self.config["vocab_size"] + 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        logger.info(f"Initialized code model with vocab size: {self.config['vocab_size']}")
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Use Kaiming initialization for better gradient flow
            torch.nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token ids of shape (batch_size, sequence_length)
            
        Returns:
            Dictionary containing:
                - logits: Token logits of shape (batch_size, sequence_length, vocab_size)
                - switch: Switch logits of shape (batch_size, sequence_length)
        """
        device = input_ids.device
        
        # Handle single sequence input
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        batch_size, seq_length = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # Transform through layers
        hidden_states = self.transformer(src=hidden_states)
        hidden_states = self.ln_f(hidden_states)
        
        # Get outputs
        outputs = self.output(hidden_states)
        
        logits = outputs[..., -1, :-1]
        switch = outputs[..., -1, -1]
        full_logits = outputs[..., :-1]
        full_switch = outputs[..., -1]
            
        return {
            'logits': logits,
            'switch': switch,
            'full_logits': full_logits,
            'full_switch': full_switch
        }

    def compute_loss(
        self, 
        batch: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with distillation and entropy-based switch prediction.
        
        Args:
            batch: Dictionary containing:
                - next_token: Target next tokens 
                - entropy: Entropy values
                - logits: Teacher model logits
            output: Dictionary containing:
                - logits: Predicted token logits
                - switch: Predicted switch values
        """
        # Get model outputs
        student_logits = output["logits"]
        switch = output["switch"]
        
        # Get batch inputs
        next_token = batch["next_token"]
        entropy = batch["entropy"]
        teacher_logits = batch["logits"]
        token_counts = batch["token_counts"]
        
        # Temperature for knowledge distillation
        temperature = 2.0
        
        # Compute reverse KL distillation loss with temperature scaling
        soft_student = F.softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            soft_student.view(-1, soft_student.size(-1)),
            soft_teacher.view(-1, soft_teacher.size(-1)),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            next_token.view(-1),
            ignore_index=-100,
            label_smoothing=0.1,
        )
        
        # Compute count-based entropy for each example
        count_entropy = torch.zeros_like(entropy)
        for i, counts in enumerate(token_counts):
            total = sum(counts.values())
            probs = [count/total for count in counts.values()]
            count_entropy[i] = -sum(p * math.log(p) for p in probs)
            
        # Combine model entropy and count-based entropy
        combined_entropy = 1.0 * entropy + 0.0 * count_entropy
        
        Create entropy-based target mask using combined entropy
        switch_target = torch.zeros_like(combined_entropy)
        switch_target[combined_entropy >= 1.2] = 1.0
        switch_target[combined_entropy < 1.2] = 0.0
        importance_weights = torch.ones_like(entropy)
        importance_weights[entropy >= 1.2] = 2.0    # Higher weight for high entropy cases (minority class)
        
        # switch_target = torch.zeros_like(entropy)
        # switch_target[entropy >= 0.6] = 1.0
        # switch_target[entropy < 0.6] = 0.0
        # importance_weights = torch.ones_like(entropy)
        # importance_weights[entropy >= 0.6] = 2.0    # Higher weight for high entropy cases (minority class)
        # mask = (entropy >= 0.4) & (entropy < 0.6)   # Create a mask for values between 0.4 and 0.6
        # importance_weights[mask] = 1.5   
        
        switch_loss = F.binary_cross_entropy_with_logits(
            switch.view(-1),
            switch_target.view(-1),
            weight=importance_weights.view(-1),
            reduction='mean'
        )

        alpha_distill = self.config["alpha_distill"]
        alpha_ce = self.config["alpha_ce"]
        alpha_switch = self.config["alpha_switch"]
        
        total_loss = (alpha_distill * distillation_loss + 
                    alpha_ce * ce_loss +
                    alpha_switch * switch_loss)
        
        return {
            'total_loss': total_loss,
            'distillation_loss': distillation_loss,
            'ce_loss': ce_loss,
            'switch_loss': switch_loss,
        }