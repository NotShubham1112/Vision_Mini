"""
Feature Extractor
Extract semantic visual embeddings using CLIP for object understanding
"""

import numpy as np
import torch
import cv2
from typing import List, Optional, Tuple
from PIL import Image
import open_clip
from config import SystemConfig


class CLIPFeatureExtractor:
    """
    Extract visual features using CLIP (Contrastive Language-Image Pre-training)
    Converts image crops into semantic embedding vectors for similarity search
    """
    
    def __init__(self, config: SystemConfig, device: Optional[str] = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model and preprocessing
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
        print(f"[FEATURE_EXTRACTOR] Initializing on {self.device}...")
    
    def initialize(self):
        """Load CLIP model"""
        if self.model is not None:
            return
        
        try:
            # Load CLIP ViT-B/32 model
            model_name = self.config.CLIP_MODEL.split('/')[-1] if hasattr(self.config, 'CLIP_MODEL') else 'ViT-B-32'
            pretrained = 'openai'
            
            print(f"[FEATURE_EXTRACTOR] Loading CLIP model: {model_name}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            self.model.eval()
            if self.device == "cpu":
                print(f"[FEATURE_EXTRACTOR] CLIP model loaded on CPU (slow)")
                print(f"[FEATURE_EXTRACTOR] WARNING: CUDA not available. Performance will be degraded.")
            else:
                print(f"[FEATURE_EXTRACTOR] CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[FEATURE_EXTRACTOR] Error loading CLIP: {e}")
            raise
    
    def extract_embedding(self, image_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract CLIP embedding from a single image crop
        
        Args:
            image_crop: BGR image (OpenCV format)
            
        Returns:
            512-dim embedding vector (normalized) or None if failed
        """
        if self.model is None:
            self.initialize()
        
        try:
            # Convert BGR to RGB
            if len(image_crop.shape) == 2:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
            elif image_crop.shape[2] == 3:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_crop)
            
            # Preprocess and extract features
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize embedding
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embedding = image_features.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            print(f"[FEATURE_EXTRACTOR] Error extracting embedding: {e}")
            return None
    
    def batch_extract(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple crops (GPU optimized)
        
        Args:
            crops: List of BGR images
            
        Returns:
            Array of shape (N, 512) with embeddings
        """
        if self.model is None:
            self.initialize()
        
        if not crops:
            return np.array([])
        
        try:
            # Preprocess all images
            pil_images = []
            for crop in crops:
                # Convert BGR to RGB
                if len(crop.shape) == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                elif crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(crop))
            
            # Stack into batch
            image_tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embeddings = image_features.cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            print(f"[FEATURE_EXTRACTOR] Error in batch extraction: {e}")
            return np.array([])
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Similarity score [0, 1] (higher = more similar)
        """
        # Normalize if not already
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Clip to [0, 1] range
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)
    
    def text_to_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Convert text description to embedding (for zero-shot classification)
        
        Args:
            text: Text description (e.g., "a photo of a toothbrush")
            
        Returns:
            512-dim embedding vector
        """
        if self.model is None:
            self.initialize()
        
        try:
            text_tokens = self.tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            embedding = text_features.cpu().numpy().flatten()
            return embedding
            
        except Exception as e:
            print(f"[FEATURE_EXTRACTOR] Error encoding text: {e}")
            return None
    
    def zero_shot_classify(self, image_crop: np.ndarray, candidate_labels: List[str]) -> Tuple[str, float]:
        """
        Zero-shot classification using CLIP
        
        Args:
            image_crop: BGR image
            candidate_labels: List of possible object names
            
        Returns:
            (best_label, confidence_score)
        """
        if self.model is None:
            self.initialize()
        
        try:
            # Get image embedding
            image_emb = self.extract_embedding(image_crop)
            if image_emb is None:
                return "unknown", 0.0
            
            # Get text embeddings for all candidates
            text_prompts = [f"a photo of a {label}" for label in candidate_labels]
            text_embeddings = []
            
            for prompt in text_prompts:
                text_emb = self.text_to_embedding(prompt)
                if text_emb is not None:
                    text_embeddings.append(text_emb)
            
            if not text_embeddings:
                return "unknown", 0.0
            
            # Compute similarities
            similarities = [self.compute_similarity(image_emb, text_emb) for text_emb in text_embeddings]
            
            # Get best match
            best_idx = np.argmax(similarities)
            best_label = candidate_labels[best_idx]
            best_score = similarities[best_idx]
            
            return best_label, best_score
            
        except Exception as e:
            print(f"[FEATURE_EXTRACTOR] Error in zero-shot classification: {e}")
            return "unknown", 0.0
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings"""
        return 512  # CLIP ViT-B/32 produces 512-dim embeddings
