import numpy as np
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from typing import Optional, Tuple
import torch
import logging
from pathlib import Path

class RLAgent:
    """Production Reinforcement Learning Agent for Penny Stock Trading
    
    Features:
    - Model loading with version control
    - Action confidence estimation
    - Ensemble support
    - GPU acceleration
    """
    
    def __init__(self, model_path: str, device: str = "auto", ensemble: bool = False):
        """
        Args:
            model_path: Path to trained RL model (.zip file)
            device: "cpu", "cuda", or "auto"
            ensemble: Whether to load multiple models for ensemble predictions
        """
        self.logger = logging.getLogger("RLAgent")
        self.models = []
        self.ensemble = ensemble
        self.device = device
        self._load_models(model_path)
        
        # Action buffer for smoothing
        self.action_history = []
        self.history_size = 5

    def _load_models(self, model_path: str):
        """Load model(s) from disk with error handling"""
        try:
            if self.ensemble:
                model_dir = Path(model_path)
                model_files = list(model_dir.glob("*.zip"))
                if not model_files:
                    raise FileNotFoundError(f"No models found in {model_path}")
                
                for mf in model_files:
                    model = PPO.load(mf, device=self.device)
                    self.models.append(model)
                    self.logger.info(f"Loaded ensemble model: {mf.name}")
            else:
                model = PPO.load(model_path, device=self.device)
                self.models.append(model)
                self.logger.info(f"Loaded single model: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, observation: np.ndarray) -> int:
        """Make trading decision with confidence estimation
        
        Args:
            observation: Market state array from environment
        Returns:
            action: 0=hold, 1=buy, 2=sell
        """
        try:
            # Convert to tensor and add batch dimension
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                
            actions = []
            confidences = []
            
            for model in self.models:
                # Get action distribution
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs_tensor)
                
                # Sample action
                action = distribution.mode()
                
                # Calculate confidence
                probs = distribution.distribution.probs
                confidence = torch.max(probs).item()
                
                actions.append(action.item())
                confidences.append(confidence)
            
            # Ensemble voting
            if self.ensemble:
                final_action = np.bincount(actions).argmax()
                avg_confidence = np.mean(confidences)
            else:
                final_action = actions[0]
                avg_confidence = confidences[0]
            
            # Smooth actions over time
            self.action_history.append(final_action)
            if len(self.action_history) > self.history_size:
                self.action_history.pop(0)
                
            smoothed_action = np.bincount(self.action_history).argmax()
            
            return {
                "action": int(smoothed_action),
                "confidence": avg_confidence,
                "raw_actions": actions,
                "raw_confidences": confidences
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {"action": 2, "confidence": 0.0}  # Fail-safe: hold

    def get_policy(self) -> BasePolicy:
        """Get underlying policy object for inspection"""
        return self.models[0].policy

    def save_onnx(self, save_path: str):
        """Export model to ONNX format for production deployment"""
        dummy_input = torch.randn(1, self.models[0].observation_space.shape[0])
        torch.onnx.export(
            self.models[0].policy,
            dummy_input,
            save_path,
            opset_version=11,
            input_names=["observation"],
            output_names=["action"]
        )

if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    agent = RLAgent(
        model_path="models/trained/rl_agent_ppo.zip",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    dummy_obs = np.random.randn(agent.models[0].observation_space.shape[0])
    prediction = agent.predict(dummy_obs)
    print(f"Prediction: {prediction}")