import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (seq_len, batch_size, embedding_dim)"""
        return x + self.pe[:x.size(0)]

class AdaptiveRouter(nn.Module):
    """
    Computes global_weight and personal_weight based on user history context.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Input to router: user context (e.g. history length, drift score, volatility)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
            nn.Softmax(dim=-1) # Output [global_weight, personal_weight]
        )
        
    def forward(self, user_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.router(user_context)
        return weights[:, 0], weights[:, 1] # global, personal

class PATFormer(nn.Module):
    """
    Personalized Adaptive Transaction Transformer (PATFormer).
    A compact causal transformer designed for CPU inference.
    """
    def __init__(
        self,
        num_categories: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 192,
        dropout: float = 0.15,
        max_seq_len: int = 64,
        use_router: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.use_router = use_router
        
        # Embeddings
        self.cat_embedding = nn.Embedding(num_categories + 1, d_model // 2)
        self.amt_embedding = nn.Linear(1, d_model // 2)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Multi-task Predictor Heads
        self.head_category = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_categories) # Task 1: next transaction category
        )
        self.head_amount = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1) # Task 2: next transaction amount
        )
        self.head_forecast = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_categories) # Task 3: future category expenditure
        )
        self.head_recurring = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1), # Task 4: recurring-payment probability
            nn.Sigmoid()
        )
        self.head_overspend = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1), # Task 5: overspend probability
            nn.Sigmoid()
        )
        
        # Personal Predictor Head (Local expert for next category/amount)
        if self.use_router:
            self.personal_head_category = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, num_categories)
            )
            self.router = AdaptiveRouter(d_model)
            
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, category: torch.Tensor, amount: torch.Tensor, user_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        category: (batch_size, seq_len)
        amount: (batch_size, seq_len, 1)
        user_context: (batch_size, d_model) - Required if use_router is True
        """
        cat_emb = self.cat_embedding(category)
        amt_emb = self.amt_embedding(amount)
        
        # Combine embeddings (batch_size, seq_len, d_model)
        x = torch.cat([cat_emb, amt_emb], dim=-1)
        
        # Add positional encoding
        # Transpose for pos_encoder which expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        seq_len = x.size(1)
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        output = self.transformer_encoder(x, mask=mask, is_causal=True)
        
        # Take the last sequence output for prediction
        last_out = output[:, -1, :] # (batch_size, d_model)
        
        # Multi-task outputs
        pred_category = self.head_category(last_out)
        pred_amount = self.head_amount(last_out)
        pred_forecast = self.head_forecast(last_out)
        pred_recurring = self.head_recurring(last_out)
        pred_overspend = self.head_overspend(last_out)
        
        if self.use_router and user_context is not None:
            personal_pred_cat = self.personal_head_category(last_out)
            w_global, w_personal = self.router(user_context)
            w_global = w_global.unsqueeze(-1)
            w_personal = w_personal.unsqueeze(-1)
            
            final_pred_cat = (w_global * pred_category) + (w_personal * personal_pred_cat)
            
            return {
                "category": final_pred_cat,
                "amount": pred_amount,
                "forecast": pred_forecast,
                "recurring": pred_recurring,
                "overspend": pred_overspend
            }
        
        return {
            "category": pred_category,
            "amount": pred_amount,
            "forecast": pred_forecast,
            "recurring": pred_recurring,
            "overspend": pred_overspend
        }
