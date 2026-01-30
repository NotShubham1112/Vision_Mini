"""
Build Embedding Database
Precompute CLIP embeddings for reference objects and build FAISS index
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from config import SystemConfig
from feature_extractor import CLIPFeatureExtractor
from dataset_manager import EmbeddingDatabase, DatasetBuilder


def build_from_directory(dataset_path: str, output_path: str):
    """
    Build embedding database from a directory of images
    
    Expected structure:
        dataset_path/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
    """
    print(f"\n{'='*60}")
    print("BUILDING EMBEDDING DATABASE FROM DIRECTORY")
    print(f"{'='*60}\n")
    
    config = SystemConfig()
    extractor = CLIPFeatureExtractor(config)
    extractor.initialize()
    
    builder = DatasetBuilder(extractor)
    db = builder.build_from_directory(dataset_path, output_path)
    
    if db:
        print(f"\n✓ Database built successfully!")
        print(f"  Location: {output_path}")
        print(f"  Stats: {db.get_stats()}")
    else:
        print("\n✗ Failed to build database")


def build_sample_database(output_path: str, num_classes: int = 20, images_per_class: int = 5):
    """
    Build a sample database with synthetic data for testing
    """
    print(f"\n{'='*60}")
    print("BUILDING SAMPLE EMBEDDING DATABASE")
    print(f"{'='*60}\n")
    
    config = SystemConfig()
    extractor = CLIPFeatureExtractor(config)
    extractor.initialize()
    
    # Common household objects
    object_classes = [
        "bottle", "cup", "phone", "laptop", "mouse", "keyboard",
        "pen", "notebook", "book", "remote", "glasses", "watch",
        "headphones", "charger", "wallet", "keys", "bag", "shoe",
        "toothbrush", "comb"
    ][:num_classes]
    
    print(f"Generating embeddings for {num_classes} classes...")
    print(f"Using CLIP text embeddings (zero-shot)\n")
    
    embeddings_list = []
    labels_list = []
    metadata_list = []
    
    for obj_class in tqdm(object_classes, desc="Processing classes"):
        for i in range(images_per_class):
            # Use text embeddings as proxy (in real use, these would be from actual images)
            text_prompt = f"a photo of a {obj_class}"
            embedding = extractor.text_to_embedding(text_prompt)
            
            if embedding is not None:
                # Add small noise to create variation
                embedding = embedding + np.random.randn(512) * 0.05
                embedding = embedding / np.linalg.norm(embedding)
                
                embeddings_list.append(embedding)
                labels_list.append(obj_class)
                metadata_list.append({
                    'class': obj_class,
                    'source': 'text_embedding',
                    'variant': i
                })
    
    # Build database
    embeddings_array = np.array(embeddings_list)
    db = EmbeddingDatabase(embedding_dim=512)
    db.build_index(embeddings_array, labels_list, metadata_list)
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    db.save_index(output_path)
    
    print(f"\n✓ Sample database created!")
    print(f"  Location: {output_path}")
    print(f"  Stats: {db.get_stats()}")
    
    return db


def test_database(db_path: str):
    """Test a built database"""
    print(f"\n{'='*60}")
    print("TESTING EMBEDDING DATABASE")
    print(f"{'='*60}\n")
    
    config = SystemConfig()
    extractor = CLIPFeatureExtractor(config)
    extractor.initialize()
    
    db = EmbeddingDatabase(embedding_dim=512)
    
    if not db.load_index(db_path):
        print("✗ Failed to load database")
        return
    
    print(f"✓ Database loaded")
    print(f"  Stats: {db.get_stats()}\n")
    
    # Test search
    test_queries = ["bottle", "laptop", "phone"]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        
        # Get text embedding
        query_emb = extractor.text_to_embedding(f"a photo of a {query}")
        
        # Search
        similarities, labels, metadata = db.search(query_emb, k=5)
        
        print(f"  Top 5 matches:")
        for i, (sim, label) in enumerate(zip(similarities, labels)):
            print(f"    {i+1}. {label} (similarity: {sim:.3f})")
        print()


def download_open_images_subset():
    """
    Download a subset of Open Images dataset
    (Placeholder - implement if needed)
    """
    print("Open Images download not yet implemented")
    print("Please use --sample mode or provide your own dataset directory")


def main():
    parser = argparse.ArgumentParser(description="Build CLIP embedding database")
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['sample', 'directory', 'open_images'],
                       help='Database building mode')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory (for directory mode)')
    parser.add_argument('--output', type=str, default='data/embeddings',
                       help='Output path for database')
    parser.add_argument('--num-classes', type=int, default=20,
                       help='Number of classes for sample mode')
    parser.add_argument('--images-per-class', type=int, default=5,
                       help='Images per class for sample mode')
    parser.add_argument('--test', action='store_true',
                       help='Test the database after building')
    
    args = parser.parse_args()
    
    print("\n" + "🗄️ "*30)
    print(" "*20 + "EMBEDDING DATABASE BUILDER")
    print("🗄️ "*30)
    
    if args.mode == 'sample':
        db = build_sample_database(args.output, args.num_classes, args.images_per_class)
    
    elif args.mode == 'directory':
        if not args.dataset:
            print("Error: --dataset required for directory mode")
            return 1
        build_from_directory(args.dataset, args.output)
    
    elif args.mode == 'open_images':
        download_open_images_subset()
        return 1
    
    if args.test:
        test_database(args.output)
    
    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
