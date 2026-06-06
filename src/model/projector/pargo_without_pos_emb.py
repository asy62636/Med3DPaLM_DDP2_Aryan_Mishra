import torch
import torch.nn as nn
from .qformer_bert import BertLMHeadModel, BertConfig

class ParGoProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Number of output tokens
        self.proj_out_num = int(getattr(config, "proj_out_num", 304))
        
        # Vision and LLM dimensions
        vision_width = int(getattr(config, "vision_hidden_size", 768))
        llm_hidden_size = int(getattr(config, "llm_hidden_size", 3584))
        
        # Load BERT config
        bert_cfg = BertConfig.from_pretrained(config.bert_type)
        bert_hidden = bert_cfg.hidden_size  # 768
        
        # Vision to BERT projection
        self.vision_proj = nn.Linear(vision_width, bert_hidden)
        
        # BERT to LLM projection  
        self.output_proj = nn.Linear(bert_hidden, llm_hidden_size)
        
        # Configure BERT for cross-attention
        bert_cfg.add_cross_attention = True
        bert_cfg.cross_attention_freq = 2
        bert_cfg.encoder_width = bert_hidden
        bert_cfg.query_length = self.proj_out_num
        bert_cfg.use_cache = False
        bert_cfg.local_query_length = [0]  # FIX: Add this line
        
        # Initialize BERT
        self.bert = BertLMHeadModel.from_pretrained(
            config.bert_type, 
            config=bert_cfg
        )
        
        # Resize token embeddings if needed
        if hasattr(config, "vocab_size") and config.vocab_size:
            self.bert.resize_token_embeddings(config.vocab_size)
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.proj_out_num, bert_hidden)
        )
        nn.init.normal_(self.query_tokens, std=0.02)
        
    def forward(self, visual_features):
        # Handle list/tuple input - take last layer features
        if isinstance(visual_features, (list, tuple)):
            visual_features = visual_features[-1]
        print("*"*20 + "Incoming visual features shape = ", visual_features.shape)
        B, N, D = visual_features.shape
        
        # Project vision features to BERT space
        print(visual_features.dtype)
        visual_features = self.vision_proj(visual_features)  # (B, N, bert_hidden)
        
        # Expand query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, proj_out_num, bert_hidden)
        
        # Create attention mask for visual tokens
        encoder_mask = torch.ones(
            B, N, 
            dtype=torch.long, 
            device=visual_features.device
        )
        
        # BERT cross-attention
        outputs = self.bert.bert(
            input_ids=None,
            attention_mask=None,
            query_embeds=query_tokens,
            encoder_hidden_states=visual_features,
            encoder_attention_mask=encoder_mask,
            return_dict=True,
            is_decoder=True,
        )
        
        # Extract query outputs
        query_output = outputs.last_hidden_state[:, :self.proj_out_num, :]
        
        # Project to LLM space
        image_features = self.output_proj(query_output)  # (B, proj_out_num, llm_hidden)
        
        return image_features
    
    def load_state_dict(self, state_dict, strict=False):
        """Flexible state dict loading"""
        # Filter out incompatible keys
        own_state = self.state_dict()
        compatible_state = {}
        
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                compatible_state[k] = v
            else:
                print(f"Skipping incompatible key: {k}")
        
        return super().load_state_dict(compatible_state, strict=False)