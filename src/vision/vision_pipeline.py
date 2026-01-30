"""
Vision Pipeline
Real-time object detection, tracking, and face recognition
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import threading
import os
import contextlib

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False
    class FaceAnalysis:
        def __init__(self, *args, **kwargs):
            print("[VISION] WARNING: InsightFace not installed. Face recognition will be disabled.")
        def prepare(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            return []

from src.core.config import SystemConfig
from src.core.shared_memory import SharedMemorySystem, TrackedPerson

# Optional reasoning imports
try:
    from src.vision.feature_extractor import CLIPFeatureExtractor
    from src.vision.dataset_manager import EmbeddingDatabase
    from src.reasoning.reasoning_engine import VisionLanguageReasoner, MultimodalFusion
    HAS_REASONING = True
except ImportError as e:
    print(f"[VISION] Reasoning modules not available: {e}")
    print("[VISION] System will run with YOLO detection only")
    HAS_REASONING = False
    CLIPFeatureExtractor = None
    EmbeddingDatabase = None
    VisionLanguageReasoner = None
    MultimodalFusion = None


class VisionPipeline:
    """
    Vision processing pipeline:
    - Capture webcam frames
    - Detect objects/people with YOLOv8
    - Track individuals with DeepSORT
    - Extract face embeddings with InsightFace
    - Match with known identities
    - Display annotated frames
    """
    
    def __init__(self, config: SystemConfig, shared_memory: SharedMemorySystem):
        self.config = config
        self.shared_memory = shared_memory
        self.running = False
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Video capture
        self.cap = None
        
        # Models
        self.detector = None
        self.grounding_detector = None # OWL-ViT
        self.tracker = None
        self.face_model = None
        
        # Reasoning components
        self.feature_extractor = None  # CLIP
        self.embedding_db = None       # FAISS
        self.reasoner = None           # LLaVA
        self.fusion = None             # Multimodal fusion
        
        # Track ID to last known name mapping
        self.track_names: Dict[int, str] = {}
        
        # Pending grounding queries
        self.grounding_query: Optional[str] = None
        
        print("[VISION] Vision pipeline initialized")
    
    def _initialize_models(self):
        """Initialize ML models (YOLOv8, DeepSORT, InsightFace)"""
        if self.detector is not None:
            return
        print("[VISION] Loading models...")
        
        try:
            # Aggressively disable progress bars and handle-locking logs
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            
            # Silence output during initialization to prevent handle issues (WinError 6)
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                # We trust devnull to handle most noise. 
                # Progress bars in background threads are the main culprit for WinError 6.

                # Initialize YOLO detector (v8m)
                self.detector = YOLO(self.config.YOLO_MODEL)
                if self.config.VISION_USE_GPU:
                    self.detector.to('cuda')
                
                # Initialize DeepSORT tracker
                self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
                
                # Initialize face recognition
                self.face_model = FaceAnalysis(name='buffalo_l')
                self.face_model.prepare(ctx_id=0 if self.config.VISION_USE_GPU else -1, det_size=(640, 640))

                # Initialize OWL-ViT for Grounding
                self.grounding_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
                self.grounding_detector = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
                if self.config.VISION_USE_GPU:
                    self.grounding_detector.to('cuda')
                
                # Initialize reasoning components
                if HAS_REASONING:
                    print("[VISION] Loading reasoning models...")
                    
                    # CLIP feature extractor
                    self.feature_extractor = CLIPFeatureExtractor(self.config)
                    self.feature_extractor.initialize()
                    
                    # Embedding database (FAISS)
                    if self.config.USE_EMBEDDING_DB:
                        self.embedding_db = EmbeddingDatabase(embedding_dim=self.config.EMBEDDING_DIM)
                        # Try to load existing index
                        if not self.embedding_db.load_index(self.config.EMBEDDING_DB_PATH):
                            print("[VISION] No embedding database found. Will use CLIP zero-shot only.")
                            self.embedding_db = None
                    
                    # Vision-Language Reasoner (LLaVA) - lazy load to save memory
                    if self.config.USE_VLM_REASONING:
                        self.reasoner = VisionLanguageReasoner(self.config)
                        # Don't initialize yet - will load on first use
                    
                    # Multimodal fusion
                    self.fusion = MultimodalFusion(self.config)
                else:
                    print("[VISION] Reasoning disabled (missing dependencies)")
            
            print(f"[VISION] ✓ YOLOv8 model loaded: {self.config.YOLO_MODEL}")
            print("[VISION] ✓ DeepSORT tracker initialized")
            print("[VISION] ✓ InsightFace model loaded")
            print("[VISION] ✓ OWL-ViT v2 loaded")
            
        except Exception as e:
            print(f"[VISION] Error loading models: {e}")
            raise
    
    def _initialize_camera(self):
        """Initialize video capture"""
        print(f"[VISION] Opening camera {self.config.CAMERA_ID}...")
        
        self.cap = cv2.VideoCapture(self.config.CAMERA_ID)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.TARGET_FPS)
        
        print("[VISION] ✓ Camera opened successfully")
    
    def run(self):
        """Main vision pipeline loop"""
        self.running = True
        
        try:
            # Initialize components
            self._initialize_models()
            self._initialize_camera()
            
            print("[VISION] Starting main loop...")
            
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("[VISION] Failed to read frame. Retrying in 1s...")
                    time.sleep(1.0)
                    continue
                
                # Process frame
                annotated_frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow('Multimodal Vision System', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                
                # Update FPS
                self._update_fps()
        
        except Exception as e:
            print(f"[VISION] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pipeline"""
        
        # 1. Detect objects and people with optimized parameters (conf=0.25, imgsz=960)
        results = self.detector.predict(
            source=frame,
            conf=self.config.YOLO_CONFIDENCE,
            imgsz=self.config.DETECTION_IMGSZ,
            iou=0.5,
            classes=self.config.YOLO_CLASSES,
            verbose=False
        )[0]
        
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            label = self.detector.names[int(class_id)]
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))

        # 2. Hybrid Grounding (OWL-ViT) - Run both temporary and permanent queries
        self._run_grounding(frame, detections)
        
        # Run permanent queries if configured
        if hasattr(self.config, 'PERMANENT_GROUNDING_QUERIES'):
            for query in self.config.PERMANENT_GROUNDING_QUERIES:
                 self._run_grounding(frame, detections, forced_query=query)
        
        # 2.5 REASONING PIPELINE - Enhance object recognition
        if self.feature_extractor is not None and self.fusion is not None:
            detections = self._apply_reasoning(frame, detections)
        
        # 3. Update tracker with detections
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = [int(v) for v in ltrb]
            class_name = track.get_det_class()
            
            obj_data = {
                'track_id': track_id,
                'bbox': bbox,
                'class': class_name,
                'confidence': track.get_det_conf() or 1.0 # Default to 1.0 if not available
            }
            tracked_objects.append(obj_data)
            
            # 3. Process each tracked person
            if class_name == 'person':
                self._process_person(frame, obj_data)
            else:
                self._process_object(obj_data)
        
        # 4. Handle identity assignment commands
        self._handle_identity_assignments(tracked_objects, frame)
        
        # 5. Annotate frame
        annotated_frame = self._annotate_frame(frame, tracked_objects)
        
        # 6. Clean up stale tracks in shared memory (shared memory handles its own state)
        self.shared_memory.remove_stale_tracks(timeout=self.config.PERSON_TIMEOUT)
        
        return annotated_frame
    
    def _process_person(self, frame: np.ndarray, obj_data: Dict[str, Any]):
        """Process detected person: extract face, match identity"""
        track_id = int(obj_data['track_id'])
        x1, y1, x2, y2 = obj_data['bbox']
        
        # Ensure bbox is within frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_region = frame[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return
        
        # Use InsightFace to detect faces within the person's bbox
        faces = self.face_model.get(face_region)
        
        if faces:
            # Take the largest face found in the person region
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embedding = face.embedding
            
            # Store face bbox relative to original frame
            # InsightFace bbox is [x1, y1, x2, y2]
            # Since we cropped the face_region from (x1, y1), we need to add offset
            # Wait, InsightFace returns bbox relative to the input image (face_region)
            # So we add the person's bbox offset
            fx1, fy1, fx2, fy2 = face.bbox
            
            # Add offset from the person crop
            person_x1, person_y1 = obj_data['bbox'][0], obj_data['bbox'][1]
            
            real_fx1 = int(fx1 + person_x1)
            real_fy1 = int(fy1 + person_y1)
            real_fx2 = int(fx2 + person_x1)
            real_fy2 = int(fy2 + person_y1)
            
            face_bbox = [real_fx1, real_fy1, real_fx2, real_fy2]
            
            # Try to match with known faces
            matched_name = self.shared_memory.find_face_match(
                embedding,
                threshold=self.config.FACE_RECOGNITION_THRESHOLD
            )
            
            if matched_name:
                self.track_names[track_id] = matched_name
            
            # Update tracked person in shared memory
            person = TrackedPerson(
                track_id=track_id,
                bbox=obj_data['bbox'],
                confidence=obj_data['confidence'],
                face_bbox=face_bbox,
                face_embedding=embedding,
                assigned_name=self.track_names.get(track_id),
                last_seen=time.time()
            )
            
            self.shared_memory.update_tracked_person(track_id, person)
    
    def _process_object(self, obj_data: Dict[str, Any]):
        """Process detected object"""
        original_label = obj_data['class']
        custom_label = self.shared_memory.get_custom_label(original_label)
        
        if custom_label:
            obj_data['display_label'] = custom_label
    
    def _run_grounding(self, frame: np.ndarray, detections: list, forced_query: Optional[str] = None):
        """Run OWL-ViT grounding for specific queries"""
        query = forced_query if forced_query else self.shared_memory.get_object_label_flag()
        if not query:
            return

        print(f"[VISION] Running grounding for: '{query}'")
        try:
            inputs = self.grounding_processor(text=[[f"a photo of a {query}"]], images=frame, return_tensors="pt")
            if self.config.VISION_USE_GPU:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.grounding_detector(**inputs)
            
            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([frame.shape[:2]]).to(inputs['pixel_values'].device)
            results = self.grounding_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.config.YOLO_CONFIDENCE)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                # box is [xmin, ymin, xmax, ymax]
                w, h = box[2] - box[0], box[3] - box[1]
                detections.append(([box[0], box[1], w, h], score.item(), query))
                print(f"[VISION] ✓ Grounded '{query}' with score {score.item():.2f}")
        except Exception as e:
            print(f"[VISION] Grounding error: {e}")
    
    def _apply_reasoning(self, frame: np.ndarray, detections: list) -> list:
        """
        Apply vision-language reasoning to enhance object recognition
        
        Pipeline:
        1. Extract crops from detections
        2. Get CLIP embeddings
        3. Search FAISS database for similar objects
        4. Optionally invoke LLaVA for low-confidence cases
        5. Fuse predictions from YOLO + CLIP + VLM
        
        Args:
            frame: Current video frame
            detections: List of (bbox, score, label) tuples from YOLO
            
        Returns:
            Enhanced detections with reasoning-based labels
        """
        enhanced_detections = []
        
        for detection in detections:
            bbox, yolo_conf, yolo_label = detection
            x, y, w, h = bbox
            
            # Skip people (handled by face recognition)
            if yolo_label == 'person':
                enhanced_detections.append(detection)
                continue
            
            # Skip very small objects
            if w < self.config.MIN_OBJECT_SIZE or h < self.config.MIN_OBJECT_SIZE:
                enhanced_detections.append(detection)
                continue
            
            try:
                # Extract crop
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # Ensure within frame bounds
                frame_h, frame_w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    enhanced_detections.append(detection)
                    continue
                
                # Get CLIP embedding
                clip_embedding = self.feature_extractor.extract_embedding(crop)
                
                if clip_embedding is None:
                    enhanced_detections.append(detection)
                    continue
                
                # Search in embedding database
                clip_label = None
                clip_score = 0.0
                
                if self.embedding_db is not None:
                    similarities, labels, metadata = self.embedding_db.search(clip_embedding, k=1)
                    if similarities:
                        clip_score = similarities[0]
                        clip_label = labels[0]
                
                # Decide whether to use VLM reasoning
                vlm_label = None
                vlm_conf = None
                
                if self.reasoner is not None and self.fusion.should_use_vlm(yolo_conf, clip_score):
                    # Only invoke expensive VLM for uncertain cases
                    if self.reasoner.enabled:
                        if not self.reasoner.model:
                            self.reasoner.initialize()
                        
                        description = self.reasoner.describe_object(crop)
                        vlm_label = self.reasoner.extract_object_name(description)
                        vlm_conf = 0.7 if vlm_label != "unknown" else 0.3
                
                # Fuse predictions
                result = self.fusion.fuse_predictions(
                    yolo_label=yolo_label,
                    yolo_conf=yolo_conf,
                    clip_label=clip_label,
                    clip_score=clip_score,
                    vlm_label=vlm_label,
                    vlm_conf=vlm_conf
                )
                
                # Update detection with fused result
                final_label = result['label']
                final_conf = result['confidence']
                
                # Store enhanced detection
                enhanced_detections.append((bbox, final_conf, final_label))
                
                # Log reasoning for interesting cases
                if result['source'] != 'YOLO' and result['source'] != 'YOLO_FALLBACK':
                    print(f"[REASONING] {yolo_label} → {final_label} ({result['source']}, {final_conf:.2f})")
                
            except Exception as e:
                print(f"[REASONING] Error processing detection: {e}")
                enhanced_detections.append(detection)
                continue
        
        return enhanced_detections

    def _handle_identity_assignments(self, tracked_objects: List[Dict[str, Any]], frame: np.ndarray):
        """Handle voice commands for identity assignment"""
        
        # Fetch pending commands from shared memory
        commands = self.shared_memory.get_pending_commands()
        for cmd in commands:
            if cmd.type == 'assign_self_identity':
                self._handle_self_id(tracked_objects, cmd.name)
            elif cmd.type == 'assign_person_identity':
                self._handle_person_id(tracked_objects, cmd.name, cmd.raw_text)
            elif cmd.type == 'assign_object_label':
                # Already handled by _run_grounding via polling object_label_flag
                # But we can also handle it here if we want to confirm
                print(f"[VISION] Command received: Label object as '{cmd.label}'")

    def _handle_self_id(self, tracked_objects, name):
        person_tracks = [t for t in tracked_objects if t['class'] == 'person']
        if person_tracks:
            central_track = self._find_central_person(person_tracks)
            if central_track:
                self._learn_identity(central_track, name)

    def _handle_person_id(self, tracked_objects, name, raw_text):
        person_tracks = [t for t in tracked_objects if t['class'] == 'person']
        if person_tracks:
            # If the user says "this person", find the most central
            if "this" in raw_text.lower() or "here" in raw_text.lower():
                target = self._find_central_person(person_tracks)
            else:
                # Default to most confident for "that person"
                target = max(person_tracks, key=lambda t: t['confidence'])
            
            if target:
                self._learn_identity(target, name)

    def _learn_identity(self, track: Dict[str, Any], name: str):
        """Extract embedding and save to shared memory"""
        track_id = int(track['track_id'])
        person = self.shared_memory.get_tracked_person(track_id)
        
        if person and person.face_embedding is not None:
            self.shared_memory.add_known_face(name, person.face_embedding)
            self.track_names[track_id] = name
            print(f"[VISION] ✓ Learned identity: {name} (Track {track_id})")

    def _find_central_person(self, person_tracks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the most central person in frame"""
        if not person_tracks:
            return None
        
        frame_center_x = self.config.FRAME_WIDTH / 2
        frame_center_y = self.config.FRAME_HEIGHT / 2
        
        def distance_to_center(track):
            bbox = track['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            return ((center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2) ** 0.5
        
        return min(person_tracks, key=distance_to_center)
    
    def _annotate_frame(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()
        
        for track in tracked_objects:
            x1, y1, x2, y2 = track['bbox']
            
            # Get label
            if track['class'] == 'person':
                track_id = int(track['track_id'])
                person = self.shared_memory.get_tracked_person(track_id)
                name = self.track_names.get(track_id, f"Person {track_id}")
                label = f"{name} ({track['confidence']:.2f})"
                color = (0, 255, 0)  # Green for people
                
                # Use face bbox if available
                if person and person.face_bbox:
                    x1, y1, x2, y2 = person.face_bbox
                # Else fall back to body bbox (already set above)
            else:
                original_label = track['class']
                custom_label = self.shared_memory.get_custom_label(original_label)
                label = custom_label or original_label
                label = f"{label} ({track['confidence']:.2f})"
                color = (255, 0, 0)  # Blue for objects
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.config.BBOX_THICKNESS)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.TEXT_SCALE, self.config.TEXT_THICKNESS
            )
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.TEXT_SCALE,
                       (255, 255, 255), self.config.TEXT_THICKNESS)
        
        # Draw FPS and stats
        known_faces_count = len(self.shared_memory.get_all_known_faces())
        stats_text = f"FPS: {self.fps:.1f} | Objects: {len(tracked_objects)} | Known Faces: {known_faces_count}"
        cv2.putText(annotated, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw latest transcription
        transcription, timestamp = self.shared_memory.get_latest_transcription()
        if transcription and (time.time() - timestamp < 3.0): # Only show for 3 seconds
            cv2.putText(annotated, f"Heard: {transcription}", (10, annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return annotated
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        
        if self.frame_count % 10 == 0: # Update every 10 frames for more responsiveness
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'tracked_persons': len(self.track_names)
        }
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
    
    def _cleanup(self):
        """Clean up resources"""
        print("[VISION] Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()


# ========== MOCK IMPLEMENTATIONS (Replace with real models) ==========

class MockYOLODetector:
    """Mock YOLO detector - replace with real YOLOv8"""
    
    def __init__(self, config):
        self.config = config
        # In real implementation: load YOLOv8 model
        # from ultralytics import YOLO
        # self.model = YOLO(config.YOLO_MODEL)
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Mock detection - returns empty list"""
        # In real implementation: run YOLO inference
        # results = self.model(frame, conf=self.config.YOLO_CONFIDENCE)
        # return parse_yolo_results(results)
        return []


class SimpleTracker:
    """Simple tracking - replace with DeepSORT"""
    
    def __init__(self, config):
        self.config = config
        self.next_id = 0
        self.tracks = {}
    
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """Mock tracking"""
        # In real implementation: use DeepSORT
        tracked = []
        
        for det in detections:
            det['track_id'] = self.next_id
            det['last_seen'] = time.time()
            self.next_id += 1
            tracked.append(det)
        
        return tracked


class MockFaceRecognizer:
    """Mock face recognizer - replace with InsightFace"""
    
    def __init__(self, config):
        self.config = config
        # In real implementation: load InsightFace model
        # from insightface.app import FaceAnalysis
        # self.app = FaceAnalysis(name=config.INSIGHTFACE_MODEL)
        # self.app.prepare(ctx_id=0 if config.USE_GPU else -1)
    
    def extract_embedding(self, face_region: np.ndarray) -> Optional[np.ndarray]:
        """Mock embedding extraction"""
        # In real implementation: extract face embedding
        # faces = self.app.get(face_region)
        # if faces:
        #     return faces[0].embedding
        # return None
        
        # Return random embedding for demo
        if face_region.size > 0:
            return np.random.randn(512).astype(np.float32)
        return None