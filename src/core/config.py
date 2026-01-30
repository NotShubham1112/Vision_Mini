"""System-wide configuration settings."""

from dataclasses import dataclass, field
from typing import List

@dataclass
class SystemConfig:
    # Quick performance tuning
    YOLO_MODEL: str = "yolov8m.pt"  # Upgrade to medium for better accuracy

    FRAME_WIDTH: int = 1280         
    FRAME_HEIGHT: int = 720
    DETECTION_IMGSZ: int = 960      # Increased for small object detection
    YOLO_CONFIDENCE: float = 0.70   # Higher threshold to filter random objects
    VISION_USE_GPU: bool = True       

    
    # Camera settings
    CAMERA_ID: int = 0
    TARGET_FPS: int = 30
    

    
    # Recognition tuning
    FACE_RECOGNITION_THRESHOLD: float = 0.75  # 0.5=lenient, 0.7=strict

    
    # Visuals
    BBOX_THICKNESS: int = 2
    TEXT_SCALE: float = 0.8
    TEXT_THICKNESS: int = 2
    
    # Timeouts
    PERSON_TIMEOUT: float = 5.0  # Seconds before a track is considered stale
    
    # Vision-Language Reasoning (NEW)
    # ================================
    
    # Feature Extraction
    CLIP_MODEL: str = "ViT-B-32"  # CLIP model for embeddings
    EMBEDDING_DIM: int = 512       # CLIP ViT-B/32 embedding dimension
    
    # Vision-Language Model
    LLAVA_MODEL: str = "llava-hf/llava-v1.6-mistral-7b-hf"  # LLaVA model
    USE_VLM_REASONING: bool = True  # Enable/disable expensive VLM reasoning
    VLM_QUANTIZATION: str = "4bit"  # 4bit, 8bit, or None
    
    # Embedding Database
    EMBEDDING_DB_PATH: str = "data/embeddings"  # Path to FAISS index
    USE_EMBEDDING_DB: bool = True  # Enable similarity search
    
    # Fusion Thresholds
    CLIP_HIGH_CONFIDENCE: float = 0.92   # High confidence CLIP match
    CLIP_MEDIUM_CONFIDENCE: float = 0.70 # Medium confidence threshold
    YOLO_HIGH_CONFIDENCE: float = 0.90   # High confidence YOLO detection
    
    # Performance Optimization
    REASONING_BATCH_SIZE: int = 1        # Process objects one at a time
    USE_REASONING_CACHE: bool = True     # Cache embeddings for static objects
    REASONING_COOLDOWN: float = 1.0      # Min seconds between VLM calls per object
    
    # Selective Reasoning (save GPU memory)
    ONLY_REASON_UNKNOWN: bool = True     # Only use VLM for low-confidence detections
    MIN_OBJECT_SIZE: int = 100           # Min bbox size (pixels) to reason about
    
    # Identity Patterns
    SELF_IDENTITY_PATTERNS: List[str] = field(default_factory=lambda: [
        r"my name is ([\w\s]+)",
        r"i am ([\w\s]+)"
    ])
    
    PERSON_IDENTITY_PATTERNS: List[str] = field(default_factory=lambda: [
        r"this person is ([\w\s]+)",
        r"that person is ([\w\s]+)",
        r"his name is ([\w\s]+)",
        r"her name is ([\w\s]+)"
    ])
    
    OBJECT_LABEL_PATTERNS: List[str] = field(default_factory=lambda: [
        r"that is a ([\w\s]+)",
        r"this is a ([\w\s]+)",
        r"label this as ([\w\s]+)"
    ])
    
    YOLO_CLASSES: List[int] = field(default_factory=lambda: [0]) # Only detect persons (class 0)

    # Custom Object Detection (Grounding Queries)
    # These will always be searched for, in addition to YOLO
    PERMANENT_GROUNDING_QUERIES: List[str] = field(default_factory=list)
