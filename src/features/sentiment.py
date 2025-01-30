from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", timeout=60)
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", timeout=60)
    
    def analyze(self, texts: List[str]) -> float:
        if not texts:
            return 0.0
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        positive_scores = probabilities[:, 0].detach().numpy()
        return float(np.mean(positive_scores))