"""
Dataset Manager
Manage reference object embeddings using FAISS for similarity search
"""

import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Dict
import faiss
from pathlib import Path


class EmbeddingDatabase:
    """
    Manage a FAISS index of object embeddings for similarity search
    Enables grounding predictions in reference datasets
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.index = None
        self.labels = []
        self.metadata = []
        
        print(f"[DATASET_MANAGER] Initialized with embedding_dim={embedding_dim}")
    
    def build_index(self, embeddings: np.ndarray, labels: List[str], metadata: Optional[List[Dict]] = None):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Array of shape (N, embedding_dim)
            labels: List of N labels corresponding to embeddings
            metadata: Optional list of metadata dicts for each embedding
        """
        if embeddings.shape[0] != len(labels):
            raise ValueError(f"Embeddings ({embeddings.shape[0]}) and labels ({len(labels)}) must have same length")
        
        print(f"[DATASET_MANAGER] Building index with {len(labels)} embeddings...")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Create FAISS index (Inner Product = Cosine Similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings_normalized.astype(np.float32))
        
        # Store labels and metadata
        self.labels = labels
        self.metadata = metadata or [{} for _ in range(len(labels))]
        
        print(f"[DATASET_MANAGER] ✓ Index built with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[str], List[Dict]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            k: Number of results to return
            
        Returns:
            (similarities, labels, metadata) - Lists of length k
        """
        if self.index is None:
            print("[DATASET_MANAGER] Warning: Index not initialized")
            return [], [], []
        
        # Normalize query
        query_normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_normalized = query_normalized.reshape(1, -1).astype(np.float32)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't request more than available
        similarities, indices = self.index.search(query_normalized, k)
        
        # Extract results
        result_similarities = similarities[0].tolist()
        result_labels = [self.labels[idx] for idx in indices[0]]
        result_metadata = [self.metadata[idx] for idx in indices[0]]
        
        return result_similarities, result_labels, result_metadata
    
    def add_embedding(self, embedding: np.ndarray, label: str, metadata: Optional[Dict] = None):
        """
        Add a single embedding to the index
        
        Args:
            embedding: Vector of shape (embedding_dim,)
            label: Label for this embedding
            metadata: Optional metadata dict
        """
        if self.index is None:
            # Initialize empty index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize and add
        embedding_normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        embedding_normalized = embedding_normalized.reshape(1, -1).astype(np.float32)
        
        self.index.add(embedding_normalized)
        self.labels.append(label)
        self.metadata.append(metadata or {})
    
    def save_index(self, path: str):
        """
        Save index to disk
        
        Args:
            path: Directory path to save index and metadata
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(path, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save labels and metadata
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'labels': self.labels,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"[DATASET_MANAGER] ✓ Index saved to {path}")
    
    def load_index(self, path: str) -> bool:
        """
        Load index from disk
        
        Args:
            path: Directory path containing index and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            index_path = os.path.join(path, "faiss_index.bin")
            if not os.path.exists(index_path):
                print(f"[DATASET_MANAGER] Index file not found: {index_path}")
                return False
            
            self.index = faiss.read_index(index_path)
            
            # Load labels and metadata
            metadata_path = os.path.join(path, "metadata.pkl")
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.labels = data['labels']
                self.metadata = data['metadata']
                self.embedding_dim = data['embedding_dim']
            
            print(f"[DATASET_MANAGER] ✓ Index loaded from {path} ({self.index.ntotal} vectors)")
            return True
            
        except Exception as e:
            print(f"[DATASET_MANAGER] Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_embeddings': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'unique_labels': len(set(self.labels)),
            'index_type': type(self.index).__name__ if self.index else None
        }


class DatasetBuilder:
    """
    Helper class to build embedding databases from image datasets
    """
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def build_from_directory(self, dataset_path: str, output_path: str, 
                            file_extensions: List[str] = ['.jpg', '.jpeg', '.png']):
        """
        Build embedding database from a directory of images
        
        Directory structure:
            dataset_path/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    img3.jpg
        
        Args:
            dataset_path: Root directory containing class subdirectories
            output_path: Where to save the index
            file_extensions: Valid image extensions
        """
        import cv2
        from pathlib import Path
        
        print(f"[DATASET_BUILDER] Building database from {dataset_path}...")
        
        embeddings_list = []
        labels_list = []
        metadata_list = []
        
        dataset_root = Path(dataset_path)
        
        # Iterate through class directories
        for class_dir in dataset_root.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            print(f"[DATASET_BUILDER] Processing class: {class_name}")
            
            # Process images in this class
            image_files = []
            for ext in file_extensions:
                image_files.extend(class_dir.glob(f"*{ext}"))
            
            for img_path in image_files:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Extract embedding
                    embedding = self.feature_extractor.extract_embedding(img)
                    if embedding is None:
                        continue
                    
                    embeddings_list.append(embedding)
                    labels_list.append(class_name)
                    metadata_list.append({
                        'source_file': str(img_path),
                        'class': class_name
                    })
                    
                except Exception as e:
                    print(f"[DATASET_BUILDER] Error processing {img_path}: {e}")
                    continue
            
            print(f"[DATASET_BUILDER] ✓ {class_name}: {len([l for l in labels_list if l == class_name])} images")
        
        # Build index
        if embeddings_list:
            embeddings_array = np.array(embeddings_list)
            db = EmbeddingDatabase(embedding_dim=embeddings_array.shape[1])
            db.build_index(embeddings_array, labels_list, metadata_list)
            db.save_index(output_path)
            
            print(f"[DATASET_BUILDER] ✓ Database built successfully: {len(labels_list)} embeddings")
            return db
        else:
            print("[DATASET_BUILDER] No valid images found")
            return None
    
    def build_from_coco_subset(self, coco_path: str, output_path: str, max_per_class: int = 50):
        """
        Build database from COCO dataset subset
        
        Args:
            coco_path: Path to COCO dataset
            output_path: Where to save index
            max_per_class: Maximum images per class
        """
        # TODO: Implement COCO dataset processing
        print("[DATASET_BUILDER] COCO dataset building not yet implemented")
        pass
