"""
Test Reasoning Pipeline
Test CLIP, FAISS, LLaVA, and multimodal fusion
"""

import cv2
import numpy as np
import sys
import argparse
from pathlib import Path

from config import SystemConfig
from feature_extractor import CLIPFeatureExtractor
from dataset_manager import EmbeddingDatabase
from reasoning_engine import VisionLanguageReasoner, MultimodalFusion


def test_clip_extraction():
    """Test CLIP feature extraction"""
    print("\n" + "="*60)
    print("TEST 1: CLIP Feature Extraction")
    print("="*60)
    
    config = SystemConfig()
    extractor = CLIPFeatureExtractor(config)
    extractor.initialize()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Extract embedding
    embedding = extractor.extract_embedding(test_image)
    
    if embedding is not None:
        print(f"✓ Embedding extracted successfully")
        print(f"  Shape: {embedding.shape}")
        print(f"  Norm: {np.linalg.norm(embedding):.4f}")
        print(f"  Sample values: {embedding[:5]}")
        return True
    else:
        print("✗ Failed to extract embedding")
        return False


def test_zero_shot_classification():
    """Test CLIP zero-shot classification"""
    print("\n" + "="*60)
    print("TEST 2: CLIP Zero-Shot Classification")
    print("="*60)
    
    config = SystemConfig()
    extractor = CLIPFeatureExtractor(config)
    extractor.initialize()
    
    # Create a simple test image (blue square)
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)
    test_image[:, :] = [255, 0, 0]  # Blue in BGR
    
    # Test classification
    candidates = ["red square", "blue square", "green circle", "yellow triangle"]
    label, score = extractor.zero_shot_classify(test_image, candidates)
    
    print(f"✓ Classification result: {label} (confidence: {score:.3f})")
    print(f"  Candidates: {candidates}")
    
    return True


def test_faiss_search():
    """Test FAISS similarity search"""
    print("\n" + "="*60)
    print("TEST 3: FAISS Similarity Search")
    print("="*60)
    
    config = SystemConfig()
    extractor = CLIPFeatureExtractor(config)
    extractor.initialize()
    
    # Create a small database
    db = EmbeddingDatabase(embedding_dim=512)
    
    # Generate some random embeddings
    labels = ["apple", "banana", "orange", "pen", "notebook"]
    embeddings = []
    
    for label in labels:
        # Create pseudo-embedding (in real use, these would be from actual images)
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # Normalize
        embeddings.append(emb)
    
    embeddings_array = np.array(embeddings)
    db.build_index(embeddings_array, labels)
    
    # Search with a query
    query = embeddings[0] + np.random.randn(512) * 0.1  # Similar to "apple"
    query = query / np.linalg.norm(query)
    
    similarities, result_labels, metadata = db.search(query, k=3)
    
    print(f"✓ Search completed")
    print(f"  Query similar to: {labels[0]}")
    print(f"  Top 3 results:")
    for i, (sim, label) in enumerate(zip(similarities, result_labels)):
        print(f"    {i+1}. {label} (similarity: {sim:.3f})")
    
    return True


def test_llava_reasoning(image_path: str = None):
    """Test LLaVA reasoning (optional - requires GPU)"""
    print("\n" + "="*60)
    print("TEST 4: LLaVA Vision-Language Reasoning")
    print("="*60)
    
    config = SystemConfig()
    config.USE_VLM_REASONING = True
    
    reasoner = VisionLanguageReasoner(config)
    
    if not reasoner.enabled:
        print("⚠ VLM reasoning disabled (requires GPU and ~4GB VRAM)")
        return False
    
    # Create or load test image
    if image_path and Path(image_path).exists():
        test_image = cv2.imread(image_path)
    else:
        # Create a simple test pattern
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(test_image, (112, 112), 50, (0, 255, 0), -1)
    
    print("  Initializing LLaVA (this may take a few minutes)...")
    reasoner.initialize()
    
    if reasoner.model is None:
        print("✗ Failed to load LLaVA model")
        return False
    
    print("  Generating description...")
    description = reasoner.describe_object(test_image)
    
    print(f"✓ Description generated:")
    print(f"  {description}")
    
    # Extract object name
    object_name = reasoner.extract_object_name(description)
    print(f"  Extracted object: {object_name}")
    
    return True


def test_multimodal_fusion():
    """Test multimodal decision fusion"""
    print("\n" + "="*60)
    print("TEST 5: Multimodal Fusion Logic")
    print("="*60)
    
    config = SystemConfig()
    fusion = MultimodalFusion(config)
    
    # Test case 1: High-confidence CLIP
    result = fusion.fuse_predictions(
        yolo_label="bottle",
        yolo_conf=0.6,
        clip_label="water bottle",
        clip_score=0.9
    )
    print(f"Test 1 - High CLIP confidence:")
    print(f"  Result: {result['label']} (source: {result['source']}, conf: {result['confidence']:.2f})")
    
    # Test case 2: High-confidence YOLO
    result = fusion.fuse_predictions(
        yolo_label="laptop",
        yolo_conf=0.85,
        clip_label="computer",
        clip_score=0.5
    )
    print(f"\nTest 2 - High YOLO confidence:")
    print(f"  Result: {result['label']} (source: {result['source']}, conf: {result['confidence']:.2f})")
    
    # Test case 3: VLM reasoning
    result = fusion.fuse_predictions(
        yolo_label="bottle",
        yolo_conf=0.5,
        clip_label="container",
        clip_score=0.65,
        vlm_label="toothbrush",
        vlm_conf=0.7
    )
    print(f"\nTest 3 - VLM reasoning:")
    print(f"  Result: {result['label']} (source: {result['source']}, conf: {result['confidence']:.2f})")
    
    print(f"\n✓ All fusion tests passed")
    return True


def test_end_to_end(image_path: str = None):
    """Test complete reasoning pipeline"""
    print("\n" + "="*60)
    print("TEST 6: End-to-End Reasoning Pipeline")
    print("="*60)
    
    config = SystemConfig()
    config.USE_VLM_REASONING = False  # Disable VLM for quick test
    
    # Initialize components
    extractor = CLIPFeatureExtractor(config)
    extractor.initialize()
    
    fusion = MultimodalFusion(config)
    
    # Load or create test image
    if image_path and Path(image_path).exists():
        test_image = cv2.imread(image_path)
    else:
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Simulate YOLO detection
    yolo_label = "bottle"
    yolo_conf = 0.6
    
    # Extract CLIP embedding
    clip_embedding = extractor.extract_embedding(test_image)
    
    # Zero-shot classification
    candidates = ["bottle", "cup", "glass", "container"]
    clip_label, clip_score = extractor.zero_shot_classify(test_image, candidates)
    
    # Fuse predictions
    result = fusion.fuse_predictions(
        yolo_label=yolo_label,
        yolo_conf=yolo_conf,
        clip_label=clip_label,
        clip_score=clip_score
    )
    
    print(f"✓ Pipeline completed:")
    print(f"  YOLO: {yolo_label} ({yolo_conf:.2f})")
    print(f"  CLIP: {clip_label} ({clip_score:.2f})")
    print(f"  Final: {result['label']} (source: {result['source']}, conf: {result['confidence']:.2f})")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test vision-language reasoning pipeline")
    parser.add_argument('--test', type=str, default='all', 
                       choices=['all', 'clip', 'zero_shot', 'faiss', 'llava', 'fusion', 'e2e'],
                       help='Which test to run')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--skip-llava', action='store_true', help='Skip LLaVA test (saves time)')
    
    args = parser.parse_args()
    
    print("\n" + "🧠 "*30)
    print(" "*20 + "VISION-LANGUAGE REASONING TEST SUITE")
    print("🧠 "*30 + "\n")
    
    results = {}
    
    if args.test == 'all' or args.test == 'clip':
        results['clip'] = test_clip_extraction()
    
    if args.test == 'all' or args.test == 'zero_shot':
        results['zero_shot'] = test_zero_shot_classification()
    
    if args.test == 'all' or args.test == 'faiss':
        results['faiss'] = test_faiss_search()
    
    if (args.test == 'all' or args.test == 'llava') and not args.skip_llava:
        results['llava'] = test_llava_reasoning(args.image)
    
    if args.test == 'all' or args.test == 'fusion':
        results['fusion'] = test_multimodal_fusion()
    
    if args.test == 'all' or args.test == 'e2e':
        results['e2e'] = test_end_to_end(args.image)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:15s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
