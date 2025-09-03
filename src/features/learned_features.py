"""
Learned Feature Extraction System
Uses contrastive learning, autoencoders, and meta-learning for adaptive feature discovery
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LearnedFeatureConfig:
    """Configuration for learned features"""
    latent_dim: int = 128
    hidden_dims: List[int] = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    temperature: float = 0.07
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]


class ContrastiveEncoder(nn.Module):
    """Encoder network for contrastive learning"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return features, projections


class ContrastiveLearner:
    """Contrastive learning for discriminative feature discovery"""
    
    def __init__(self, config: LearnedFeatureConfig):
        self.config = config
        self.encoder = None
        self.device = torch.device(config.device)
        self.training_history = []
        
    def create_positive_pairs(self, X: np.ndarray, 
                             labels: Optional[np.ndarray] = None,
                             augmentation_fn: Optional[callable] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create positive pairs for contrastive learning
        
        Args:
            X: Feature matrix
            labels: Optional labels for supervised contrastive
            augmentation_fn: Function to create augmented views
        """
        X_tensor = torch.FloatTensor(X)
        
        if augmentation_fn is not None:
            # Create augmented views
            X_aug = augmentation_fn(X)
            X_aug_tensor = torch.FloatTensor(X_aug)
        else:
            # Default augmentation: add small noise
            noise = np.random.normal(0, 0.01, X.shape)
            X_aug_tensor = torch.FloatTensor(X + noise)
            
        return X_tensor, X_aug_tensor
    
    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor, 
                    temperature: float = 0.07) -> torch.Tensor:
        """
        NT-Xent loss for contrastive learning
        
        Args:
            z_i: First set of projections
            z_j: Second set of projections
            temperature: Temperature parameter
        """
        batch_size = z_i.size(0)
        
        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.mm(representations, representations.t())
        
        # Create mask for positive pairs
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=self.device)
        mask[:batch_size, batch_size:].fill_diagonal_(True)
        mask[batch_size:, :batch_size].fill_diagonal_(True)
        
        # Compute loss
        positives = similarity_matrix[mask].view(batch_size * 2, -1)
        negatives = similarity_matrix[~mask].view(batch_size * 2, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=self.device)
        
        loss = F.cross_entropy(logits / temperature, labels)
        return loss
    
    def fit(self, X: np.ndarray, labels: Optional[np.ndarray] = None,
           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Train contrastive encoder
        
        Args:
            X: Training features
            labels: Optional labels
            validation_data: Optional validation set
        """
        input_dim = X.shape[1]
        
        # Initialize encoder
        self.encoder = ContrastiveEncoder(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.latent_dim
        ).to(self.device)
        
        optimizer = optim.Adam(self.encoder.parameters(), lr=self.config.learning_rate)
        
        # Create data loader
        X_i, X_j = self.create_positive_pairs(X, labels)
        dataset = TensorDataset(X_i, X_j)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        logger.info(f"Training contrastive encoder for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch_i, batch_j in dataloader:
                batch_i = batch_i.to(self.device)
                batch_j = batch_j.to(self.device)
                
                # Forward pass
                _, proj_i = self.encoder(batch_i)
                _, proj_j = self.encoder(batch_j)
                
                # Compute loss
                loss = self.nt_xent_loss(proj_i, proj_j, self.config.temperature)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            self.training_history.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using trained encoder"""
        self.encoder.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            features, _ = self.encoder(X_tensor)
            
        return features.cpu().numpy()


class AutoEncoder(nn.Module):
    """Autoencoder for unsupervised feature learning"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror architecture)
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        return self.encoder(x)


class VariationalAutoEncoder(nn.Module):
    """Variational Autoencoder for probabilistic feature learning"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


class MetaLearner:
    """Meta-learning for few-shot feature adaptation"""
    
    def __init__(self, config: LearnedFeatureConfig):
        self.config = config
        self.meta_model = None
        self.device = torch.device(config.device)
        self.task_embeddings = {}
        
    def create_meta_model(self, input_dim: int) -> nn.Module:
        """Create meta-learning model"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.latent_dim)
        )
    
    def inner_loop_update(self, model: nn.Module, support_x: torch.Tensor, 
                         support_y: torch.Tensor, inner_lr: float = 0.01) -> nn.Module:
        """
        Inner loop update for MAML-style meta-learning
        
        Args:
            model: Current model
            support_x: Support set features
            support_y: Support set labels
            inner_lr: Inner loop learning rate
        """
        # Clone model for task-specific adaptation
        adapted_model = type(model)(model.in_features, model.out_features)
        adapted_model.load_state_dict(model.state_dict())
        
        # Compute loss on support set
        pred = adapted_model(support_x)
        loss = F.mse_loss(pred, support_y)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, adapted_model.parameters())
        
        # Update parameters
        with torch.no_grad():
            for param, grad in zip(adapted_model.parameters(), grads):
                param -= inner_lr * grad
                
        return adapted_model
    
    def adapt_to_task(self, support_features: np.ndarray, 
                     support_labels: np.ndarray,
                     num_adaptation_steps: int = 5) -> np.ndarray:
        """
        Adapt features to new task with few examples
        
        Args:
            support_features: Few-shot examples
            support_labels: Labels for examples
            num_adaptation_steps: Number of adaptation steps
        """
        if self.meta_model is None:
            input_dim = support_features.shape[1]
            self.meta_model = self.create_meta_model(input_dim).to(self.device)
            
        support_x = torch.FloatTensor(support_features).to(self.device)
        support_y = torch.FloatTensor(support_labels).to(self.device)
        
        # Adapt model
        adapted_model = self.meta_model
        for _ in range(num_adaptation_steps):
            adapted_model = self.inner_loop_update(adapted_model, support_x, support_y)
            
        # Extract adapted features
        with torch.no_grad():
            adapted_features = adapted_model(support_x).cpu().numpy()
            
        return adapted_features


class OnlineFeatureRefiner:
    """Online feature refinement as more models are analyzed"""
    
    def __init__(self, initial_dim: int, memory_size: int = 1000):
        self.feature_dim = initial_dim
        self.memory_size = memory_size
        
        # Experience replay buffer
        self.feature_memory = []
        self.label_memory = []
        
        # Online statistics
        self.running_mean = np.zeros(initial_dim)
        self.running_cov = np.eye(initial_dim)
        self.n_samples = 0
        
        # Feature importance scores
        self.importance_scores = np.ones(initial_dim) / initial_dim
        
    def update(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
              importance_feedback: Optional[np.ndarray] = None):
        """
        Update feature statistics and importance online
        
        Args:
            features: New feature observations
            labels: Optional labels
            importance_feedback: Optional importance scores
        """
        # Update running statistics
        batch_size = features.shape[0]
        
        # Incremental mean update
        delta = features - self.running_mean
        self.running_mean += np.sum(delta, axis=0) / (self.n_samples + batch_size)
        
        # Incremental covariance update (simplified)
        if self.n_samples > 0:
            alpha = batch_size / (self.n_samples + batch_size)
            batch_cov = np.cov(features.T)
            self.running_cov = (1 - alpha) * self.running_cov + alpha * batch_cov
            
        self.n_samples += batch_size
        
        # Update experience replay
        for i in range(batch_size):
            if len(self.feature_memory) >= self.memory_size:
                # Remove oldest
                self.feature_memory.pop(0)
                if labels is not None:
                    self.label_memory.pop(0)
                    
            self.feature_memory.append(features[i])
            if labels is not None:
                self.label_memory.append(labels[i])
                
        # Update importance scores
        if importance_feedback is not None:
            # Exponential moving average
            alpha = 0.1
            self.importance_scores = (1 - alpha) * self.importance_scores + alpha * importance_feedback
            
    def refine_features(self, features: np.ndarray) -> np.ndarray:
        """
        Refine features using learned statistics
        
        Args:
            features: Input features
        """
        # Standardize using running statistics
        refined = (features - self.running_mean) / (np.sqrt(np.diag(self.running_cov)) + 1e-8)
        
        # Weight by importance
        refined = refined * self.importance_scores
        
        return refined
    
    def get_top_features(self, n: int = 50) -> np.ndarray:
        """Get indices of top n important features"""
        return np.argsort(self.importance_scores)[-n:][::-1]


class LearnedFeatures:
    """Main interface for learned feature extraction"""
    
    def __init__(self, config: Optional[LearnedFeatureConfig] = None):
        self.config = config or LearnedFeatureConfig()
        
        # Initialize learners
        self.contrastive_learner = ContrastiveLearner(self.config)
        self.autoencoder = None
        self.vae = None
        self.meta_learner = MetaLearner(self.config)
        self.online_refiner = None
        
        # Feature cache
        self.learned_features_cache = {}
        
    def learn_contrastive_features(self, X: np.ndarray, 
                                  labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Learn features using contrastive learning"""
        self.contrastive_learner.fit(X, labels)
        return self.contrastive_learner.transform(X)
    
    def learn_autoencoder_features(self, X: np.ndarray, 
                                  use_variational: bool = False) -> np.ndarray:
        """Learn features using autoencoder"""
        input_dim = X.shape[1]
        
        if use_variational:
            self.vae = VariationalAutoEncoder(
                input_dim=input_dim,
                hidden_dims=self.config.hidden_dims,
                latent_dim=self.config.latent_dim
            ).to(self.config.device)
            model = self.vae
        else:
            self.autoencoder = AutoEncoder(
                input_dim=input_dim,
                hidden_dims=self.config.hidden_dims,
                latent_dim=self.config.latent_dim
            ).to(self.config.device)
            model = self.autoencoder
            
        # Train autoencoder
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        X_tensor = torch.FloatTensor(X).to(self.config.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        logger.info(f"Training {'VAE' if use_variational else 'AutoEncoder'}")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                batch = batch[0].to(self.config.device)
                
                if use_variational:
                    reconstructed, mu, logvar = model(batch)
                    # VAE loss
                    recon_loss = F.mse_loss(reconstructed, batch)
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.01 * kld_loss
                else:
                    reconstructed, _ = model(batch)
                    loss = F.mse_loss(reconstructed, batch)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            if epoch % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                
        # Extract learned features
        model.eval()
        with torch.no_grad():
            if use_variational:
                features, _ = model.encode(X_tensor)
            else:
                features = model.encode(X_tensor)
                
        return features.cpu().numpy()
    
    def adapt_features_few_shot(self, support_features: np.ndarray,
                               support_labels: np.ndarray,
                               query_features: np.ndarray) -> np.ndarray:
        """
        Adapt features for few-shot learning
        
        Args:
            support_features: Few examples from new task
            support_labels: Labels for support set
            query_features: Features to adapt
        """
        # First adapt to support set
        adapted_support = self.meta_learner.adapt_to_task(
            support_features, support_labels
        )
        
        # Then transform query features
        if self.meta_learner.meta_model is not None:
            query_tensor = torch.FloatTensor(query_features).to(self.config.device)
            with torch.no_grad():
                adapted_query = self.meta_learner.meta_model(query_tensor).cpu().numpy()
        else:
            adapted_query = query_features
            
        return adapted_query
    
    def initialize_online_refiner(self, initial_features: np.ndarray):
        """Initialize online feature refiner"""
        feature_dim = initial_features.shape[1]
        self.online_refiner = OnlineFeatureRefiner(feature_dim)
        self.online_refiner.update(initial_features)
        
    def refine_features_online(self, features: np.ndarray,
                              labels: Optional[np.ndarray] = None,
                              importance_feedback: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Refine features using online learning
        
        Args:
            features: New features to refine
            labels: Optional labels
            importance_feedback: Optional importance scores
        """
        if self.online_refiner is None:
            self.initialize_online_refiner(features)
            
        # Update refiner with new observations
        self.online_refiner.update(features, labels, importance_feedback)
        
        # Return refined features
        return self.online_refiner.refine_features(features)
    
    def extract_all_learned_features(self, X: np.ndarray,
                                    labels: Optional[np.ndarray] = None,
                                    methods: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract features using all learned methods
        
        Args:
            X: Input features
            labels: Optional labels
            methods: List of methods to use
        """
        if methods is None:
            methods = ['contrastive', 'autoencoder', 'vae']
            
        all_features = {}
        
        if 'contrastive' in methods:
            try:
                features = self.learn_contrastive_features(X, labels)
                all_features['contrastive'] = features
            except Exception as e:
                logger.warning(f"Contrastive learning failed: {e}")
                
        if 'autoencoder' in methods:
            try:
                features = self.learn_autoencoder_features(X, use_variational=False)
                all_features['autoencoder'] = features
            except Exception as e:
                logger.warning(f"Autoencoder learning failed: {e}")
                
        if 'vae' in methods:
            try:
                features = self.learn_autoencoder_features(X, use_variational=True)
                all_features['vae'] = features
            except Exception as e:
                logger.warning(f"VAE learning failed: {e}")
                
        return all_features
    
    def save_models(self, directory: str):
        """Save all learned models"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(directory / 'config.json', 'w') as f:
            json.dump({
                'latent_dim': self.config.latent_dim,
                'hidden_dims': self.config.hidden_dims,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.num_epochs,
                'temperature': self.config.temperature,
                'device': self.config.device
            }, f, indent=2)
            
        # Save models
        if self.contrastive_learner.encoder is not None:
            torch.save(self.contrastive_learner.encoder.state_dict(),
                      directory / 'contrastive_encoder.pth')
                      
        if self.autoencoder is not None:
            torch.save(self.autoencoder.state_dict(),
                      directory / 'autoencoder.pth')
                      
        if self.vae is not None:
            torch.save(self.vae.state_dict(),
                      directory / 'vae.pth')
                      
        logger.info(f"Saved learned models to {directory}")
        
    def load_models(self, directory: str):
        """Load learned models"""
        directory = Path(directory)
        
        # Load config
        if (directory / 'config.json').exists():
            with open(directory / 'config.json', 'r') as f:
                config_data = json.load(f)
                self.config = LearnedFeatureConfig(**config_data)
                
        # Note: Model loading would require knowing input dimensions
        # This is typically done when transform() is called with actual data
        
        logger.info(f"Loaded model configuration from {directory}")