"""
Reasoning Engine
Vision-Language reasoning using LLaVA and multimodal decision fusion
"""

import numpy as np
import torch
import cv2
from typing import List, Optional, Dict, Tuple, Any
from PIL import Image
import re
from transformers import AutoTokenizer, BitsAndBytesConfig
from config import SystemConfig


class VisionLanguageReasoner:
    """
    Vision-Language Model for object reasoning and description
    Uses LLaVA (Large Language and Vision Assistant) for multimodal understanding
    """
    
    def __init__(self, config: SystemConfig, device: Optional[str] = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.processor = None
        self.enabled = getattr(config, 'USE_VLM_REASONING', True)
        
        print(f"[REASONING_ENGINE] Initializing VLM on {self.device}...")
    
    def initialize(self):
        """Load LLaVA model with 4-bit quantization"""
        if self.model is not None or not self.enabled:
            return
        
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            model_id = getattr(self.config, 'LLAVA_MODEL', 'llava-hf/llava-v1.6-mistral-7b-hf')
            
            print(f"[REASONING_ENGINE] Loading LLaVA model: {model_id}")
            print("[REASONING_ENGINE] This may take a few minutes on first run...")
            
            # 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load processor
            self.processor = LlavaNextProcessor.from_pretrained(model_id)
            
            # Load model with quantization
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            print(f"[REASONING_ENGINE] ✓ LLaVA model loaded successfully (4-bit)")
            
        except Exception as e:
            print(f"[REASONING_ENGINE] Error loading LLaVA: {e}")
            print("[REASONING_ENGINE] VLM reasoning will be disabled")
            self.enabled = False
    
    def describe_object(self, image_crop: np.ndarray, prompt: Optional[str] = None) -> str:
        """
        Generate natural language description of an object
        
        Args:
            image_crop: BGR image of detected object
            prompt: Optional custom prompt (uses default if None)
            
        Returns:
            Natural language description
        """
        if not self.enabled or self.model is None:
            self.initialize()
            if not self.enabled:
                return ""
        
        try:
            # Convert BGR to RGB
            if len(image_crop.shape) == 2:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
            elif image_crop.shape[2] == 3:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL
            pil_image = Image.fromarray(image_crop)
            
            # Default prompt for object description
            if prompt is None:
                prompt = """[INST] <image>\nAnalyze this object carefully. Describe:
1. Its shape and structure
2. Color and material
3. Any visible text or logos
4. What this object most likely is

Be specific and concise. End with your best guess of the object name. [/INST]"""
            
            # Process inputs
            inputs = self.processor(prompt, pil_image, return_tensors="pt").to(self.device)
            
            # Generate description
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=0.2
                )
            
            # Decode output
            description = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the response (remove prompt)
            if "[/INST]" in description:
                description = description.split("[/INST]")[-1].strip()
            
            return description
            
        except Exception as e:
            print(f"[REASONING_ENGINE] Error generating description: {e}")
            return ""
    
    def extract_object_name(self, description: str) -> str:
        """
        Extract the most likely object name from a description
        
        Args:
            description: Natural language description from VLM
            
        Returns:
            Extracted object name (lowercase)
        """
        if not description:
            return "unknown"
        
        # Look for common patterns
        patterns = [
            r"(?:this (?:is|appears to be) (?:a|an) )([a-z\s]+)",
            r"(?:likely (?:a|an) )([a-z\s]+)",
            r"(?:most likely )([a-z\s]+)",
            r"(?:object (?:is|appears to be) (?:a|an) )([a-z\s]+)",
        ]
        
        description_lower = description.lower()
        
        for pattern in patterns:
            match = re.search(pattern, description_lower)
            if match:
                object_name = match.group(1).strip()
                # Clean up
                object_name = re.sub(r'\s+', ' ', object_name)
                return object_name
        
        # Fallback: Take last sentence and extract nouns
        sentences = description.split('.')
        if sentences:
            last_sentence = sentences[-1].lower()
            # Simple noun extraction (words after "a" or "an")
            match = re.search(r'\b(?:a|an)\s+([a-z\s]+)', last_sentence)
            if match:
                return match.group(1).strip()
        
        return "unknown"
    
    def quick_classify(self, image_crop: np.ndarray, candidates: List[str]) -> Tuple[str, float]:
        """
        Quick classification among candidate labels
        
        Args:
            image_crop: BGR image
            candidates: List of possible object names
            
        Returns:
            (best_label, confidence)
        """
        if not self.enabled or not candidates:
            return "unknown", 0.0
        
        try:
            # Create a prompt asking to choose from candidates
            candidates_str = ", ".join(candidates)
            prompt = f"""[INST] <image>\nWhat is this object? Choose the best match from: {candidates_str}
Answer with just the object name. [/INST]"""
            
            description = self.describe_object(image_crop, prompt)
            
            # Find which candidate appears in the response
            description_lower = description.lower()
            for candidate in candidates:
                if candidate.lower() in description_lower:
                    return candidate, 0.8  # Moderate confidence
            
            return candidates[0], 0.3  # Low confidence fallback
            
        except Exception as e:
            print(f"[REASONING_ENGINE] Error in quick classification: {e}")
            return "unknown", 0.0


class MultimodalFusion:
    """
    Fuse predictions from multiple sources (YOLO, CLIP, VLM)
    Implements confidence-based decision logic
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Thresholds
        self.clip_high_conf = getattr(config, 'CLIP_HIGH_CONFIDENCE', 0.85)
        self.clip_medium_conf = getattr(config, 'CLIP_MEDIUM_CONFIDENCE', 0.6)
        self.yolo_high_conf = getattr(config, 'YOLO_HIGH_CONFIDENCE', 0.75)
        
        print("[FUSION] Multimodal fusion initialized")
    
    def fuse_predictions(self, 
                        yolo_label: str, 
                        yolo_conf: float,
                        clip_label: Optional[str] = None,
                        clip_score: Optional[float] = None,
                        vlm_label: Optional[str] = None,
                        vlm_conf: Optional[float] = None) -> Dict[str, Any]:
        """
        Fuse predictions from multiple sources
        
        Args:
            yolo_label: YOLO detection label
            yolo_conf: YOLO confidence [0, 1]
            clip_label: CLIP similarity search result
            clip_score: CLIP similarity score [0, 1]
            vlm_label: VLM reasoning result
            vlm_conf: VLM confidence [0, 1]
            
        Returns:
            {
                'label': final label,
                'confidence': final confidence,
                'source': which model was used,
                'all_predictions': dict of all inputs
            }
        """
        
        # Store all predictions for transparency
        all_predictions = {
            'yolo': {'label': yolo_label, 'confidence': yolo_conf},
            'clip': {'label': clip_label, 'confidence': clip_score} if clip_label else None,
            'vlm': {'label': vlm_label, 'confidence': vlm_conf} if vlm_label else None
        }
        
        # Decision logic (priority order)
        
        # 1. High-confidence CLIP match (most reliable for known objects)
        if clip_score is not None and clip_score >= self.clip_high_conf:
            return {
                'label': clip_label,
                'confidence': clip_score,
                'source': 'CLIP',
                'all_predictions': all_predictions
            }
        
        # 2. High-confidence YOLO (good for common objects)
        if yolo_conf >= self.yolo_high_conf:
            return {
                'label': yolo_label,
                'confidence': yolo_conf,
                'source': 'YOLO',
                'all_predictions': all_predictions
            }
        
        # 3. Medium-confidence CLIP + VLM reasoning
        if clip_score is not None and clip_score >= self.clip_medium_conf:
            # If we have VLM reasoning, prefer it for disambiguation
            if vlm_label and vlm_label != "unknown":
                return {
                    'label': vlm_label,
                    'confidence': (clip_score + vlm_conf) / 2 if vlm_conf else clip_score,
                    'source': 'VLM+CLIP',
                    'all_predictions': all_predictions
                }
            else:
                return {
                    'label': clip_label,
                    'confidence': clip_score,
                    'source': 'CLIP',
                    'all_predictions': all_predictions
                }
        
        # 4. VLM reasoning alone (for unknown objects)
        if vlm_label and vlm_label != "unknown" and vlm_conf and vlm_conf > 0.5:
            return {
                'label': vlm_label,
                'confidence': vlm_conf,
                'source': 'VLM',
                'all_predictions': all_predictions
            }
        
        # 5. Fallback to YOLO (even with low confidence)
        return {
            'label': yolo_label,
            'confidence': yolo_conf,
            'source': 'YOLO_FALLBACK',
            'all_predictions': all_predictions
        }
    
    def compute_ensemble_confidence(self, predictions: List[Dict[str, Any]]) -> float:
        """
        Compute ensemble confidence from multiple predictions
        
        Args:
            predictions: List of prediction dicts with 'label' and 'confidence'
            
        Returns:
            Ensemble confidence score
        """
        if not predictions:
            return 0.0
        
        # Check agreement
        labels = [p['label'] for p in predictions if p.get('label')]
        if not labels:
            return 0.0
        
        # If all agree, boost confidence
        if len(set(labels)) == 1:
            confidences = [p['confidence'] for p in predictions if p.get('confidence')]
            return min(1.0, np.mean(confidences) * 1.2)  # 20% boost for agreement
        
        # If disagree, take max confidence
        confidences = [p['confidence'] for p in predictions if p.get('confidence')]
        return max(confidences) if confidences else 0.0
    
    def should_use_vlm(self, yolo_conf: float, clip_score: Optional[float]) -> bool:
        """
        Decide whether to invoke expensive VLM reasoning
        
        Args:
            yolo_conf: YOLO confidence
            clip_score: CLIP similarity score
            
        Returns:
            True if VLM reasoning should be used
        """
        # Use VLM if both YOLO and CLIP have low confidence
        if yolo_conf < self.yolo_high_conf:
            if clip_score is None or clip_score < self.clip_medium_conf:
                return True
        
        return False
