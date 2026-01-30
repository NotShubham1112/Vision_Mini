"""Shared-memory and data exchange utilities."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

@dataclass
class TrackedPerson:
    """Data for a tracked person in the vision pipeline."""
    track_id: int
    bbox: List[int]
    confidence: float
    face_bbox: Optional[List[int]] = None
    face_embedding: Optional[np.ndarray] = None
    assigned_name: Optional[str] = None
    last_seen: float = field(default_factory=time.time)

@dataclass
class VoiceCommand:
    """Structure for a parsed voice command."""
    type: str  # 'assign_self_identity', 'assign_person_identity', 'assign_object_label'
    name: Optional[str] = None
    label: Optional[str] = None
    raw_text: str = ""
    timestamp: float = field(default_factory=time.time)

class SharedMemorySystem:
    """
    Central data store for inter-pipeline communication.
    Uses thread-safe locking for concurrent access.
    """

    def __init__(self):
        self._lock = Lock()
        self._tracked_persons: Dict[int, TrackedPerson] = {}
        self._known_faces: Dict[str, np.ndarray] = {}  # name -> embedding
        self._custom_object_labels: Dict[str, str] = {}  # original -> custom
        self._latest_transcription: Tuple[Optional[str], float] = (None, 0.0)
        self._pending_commands: List[VoiceCommand] = []
        
        # Flags for the vision pipeline to process
        self._self_identity_flag: Optional[str] = None
        self._target_identity_flag: Optional[str] = None
        self._object_label_flag: Optional[str] = None

    # Tracked Persons
    def update_tracked_person(self, track_id: int, person: TrackedPerson):
        with self._lock:
            self._tracked_persons[track_id] = person

    def get_tracked_person(self, track_id: int) -> Optional[TrackedPerson]:
        with self._lock:
            return self._tracked_persons.get(track_id)

    def remove_stale_tracks(self, timeout: float):
        current_time = time.time()
        with self._lock:
            stale_ids = [
                tid for tid, p in self._tracked_persons.items()
                if current_time - p.last_seen > timeout
            ]
            for tid in stale_ids:
                del self._tracked_persons[tid]

    # Face Recognition
    def add_known_face(self, name: str, embedding: np.ndarray):
        with self._lock:
            self._known_faces[name] = embedding

    def find_face_match(self, embedding: np.ndarray, threshold: float) -> Optional[str]:
        """Simple cosine similarity matching"""
        with self._lock:
            if not self._known_faces:
                return None
            
            best_match = None
            max_sim = -1.0
            
            # Normalize embedding
            norm_emb = embedding / np.linalg.norm(embedding)
            
            for name, known_emb in self._known_faces.items():
                norm_known = known_emb / np.linalg.norm(known_emb)
                similarity = np.dot(norm_emb, norm_known)
                
                if similarity > threshold and similarity > max_sim:
                    max_sim = similarity
                    best_match = name
            
            return best_match

    def get_all_known_faces(self) -> List[str]:
        with self._lock:
            return list(self._known_faces.keys())

    # Object Labels
    def add_custom_object_label(self, original: str, custom: str):
        with self._lock:
            self._custom_object_labels[original] = custom

    def get_custom_label(self, original: str) -> Optional[str]:
        with self._lock:
            return self._custom_object_labels.get(original)

    # Transcription and Commands
    def update_transcription(self, text: str):
        with self._lock:
            self._latest_transcription = (text, time.time())

    def get_latest_transcription(self) -> Tuple[Optional[str], float]:
        with self._lock:
            return self._latest_transcription

    def add_command(self, command: VoiceCommand):
        with self._lock:
            self._pending_commands.append(command)
            # Route flags based on command type
            if command.type == 'assign_self_identity':
                self._self_identity_flag = command.name
            elif command.type == 'assign_person_identity':
                self._target_identity_flag = command.name
            elif command.type == 'assign_object_label':
                self._object_label_flag = command.label

    # Command Flags (Stateful consumption)
    def get_pending_commands(self) -> List[VoiceCommand]:
        """Get and clear the command queue"""
        with self._lock:
            commands = self._pending_commands.copy()
            self._pending_commands.clear()
            return commands

    def get_self_identity_flag(self) -> Optional[str]:
        with self._lock:
            val = self._self_identity_flag
            self._self_identity_flag = None
            return val

    def get_target_identity_flag(self) -> Optional[str]:
        with self._lock:
            val = self._target_identity_flag
            self._target_identity_flag = None
            return val

    def get_object_label_flag(self) -> Optional[str]:
        with self._lock:
            val = self._object_label_flag
            self._object_label_flag = None
            return val
