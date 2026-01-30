"""
Audio Pipeline
Real-time speech recognition and command parsing
"""

import threading
import time
import re
import queue
import os
import contextlib
from typing import Optional, Dict, Any, List
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from src.core.config import SystemConfig
from src.core.shared_memory import SharedMemorySystem, VoiceCommand
from src.core.interpreter import VisionCommandInterpreter


class AudioPipeline:
    """
    Audio processing pipeline:
    - Continuously record microphone input
    - Detect voice activity
    - Transcribe speech with Faster-Whisper
    - Parse commands with NLP
    - Send commands to shared memory
    """
    
    def __init__(self, config: SystemConfig, shared_memory: SharedMemorySystem):
        self.config = config
        self.shared_memory = shared_memory
        self.running = False
        
        # Audio buffer
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Speech recognition model
        self.speech_model = None
        
        # Command cooldown
        self.last_command_time = 0
        
        # Statistics
        self.transcriptions_count = 0
        self.commands_parsed = 0
        
        # NLP Interpreter
        self.interpreter = VisionCommandInterpreter()
        
        print("[AUDIO] Audio pipeline initialized")
    
    def _initialize_models(self):
        """Initialize speech recognition model"""
        if self.speech_model is not None:
            return
        print(f"[AUDIO] Loading Whisper model ({self.config.WHISPER_MODEL})...")
        
        try:
            # Silence stdout/stderr during initialization in background thread to avoid WinError 6
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                # Using a safer way to disable tqdm if possible
                try:
                    import tqdm
                    # This is slightly safer than monkeypatching __init__ directly
                    if hasattr(tqdm, 'tqdm'):
                        from functools import partial
                        tqdm.tqdm = partial(tqdm.tqdm, disable=True)
                except ImportError:
                    pass

                self.speech_model = WhisperModel(
                    self.config.WHISPER_MODEL,
                    device="cuda" if self.config.AUDIO_USE_GPU else "cpu",
                    compute_type=self.config.WHISPER_COMPUTE_TYPE
                )
            print("[AUDIO] ✓ Speech recognition model loaded")
            
        except Exception as e:
            print(f"[AUDIO] Error loading models: {e}")
            raise
    
    def run(self):
        """Main audio pipeline loop"""
        self.running = True
        
        try:
            # Initialize models
            self._initialize_models()
            
            print("[AUDIO] Starting audio capture...")
            
            # Start audio capture thread
            capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
            capture_thread.start()
            
            print("[AUDIO] Starting main processing loop...")
            
            # Main processing loop
            while self.running:
                try:
                    # Get audio chunk from queue (with timeout)
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                    
                    # Process audio
                    self._process_audio(audio_chunk)
                    
                except queue.Empty:
                    continue
                
                except Exception as e:
                    print(f"[AUDIO] Error processing audio: {e}")
        
        except Exception as e:
            print(f"[AUDIO] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
    
    def _capture_audio(self):
        """Capture audio from microphone (runs in separate thread)"""
        try:
            # Try multiple devices if default fails
            input_device = None
            try:
                device_info = sd.query_devices(kind='input')
                input_device = device_info['index']
                print(f"[AUDIO] Attempting default input device: {device_info['name']} (Index {input_device})")
            except:
                print("[AUDIO] No default input device found")

            def callback(indata, frames, time, status):
                if status:
                    try:
                        print(f"[AUDIO] Status: {status}")
                    except OSError:
                        pass
                
                # Resample to 16000Hz if necessary
                data = indata.copy().flatten()
                if self.current_samplerate != 16000:
                    # Linearly resample using numpy.interp
                    duration = len(data) / self.current_samplerate
                    target_length = int(duration * 16000)
                    xp = np.linspace(0, duration, len(data))
                    x = np.linspace(0, duration, target_length)
                    data = np.interp(x, xp, data).astype(np.float32)
                
                # Put data into the queue
                self.audio_queue.put(data)

            # Try multiple devices and host APIs if default fails
            stream = None
            self.current_samplerate = 16000 # Default target
            
            host_apis = sd.query_hostapis()
            wasapi_idx = next((i for i, h in enumerate(host_apis) if 'WASAPI' in h['name']), None)
            mme_idx = next((i for i, h in enumerate(host_apis) if 'MME' in h['name']), None)

            def try_device(dev_idx):
                if dev_idx is None: return None
                try:
                    dev_info = sd.query_devices(dev_idx)
                    device_name = dev_info['name']
                    native_rate = int(dev_info['default_samplerate'])
                    
                    # Samplerates to try: 16000 first, then native, then common ones
                    samplerates_to_try = [16000]
                    if native_rate != 16000 and native_rate > 0:
                        samplerates_to_try.append(native_rate)
                    for r in [44100, 48000]:
                        if r not in samplerates_to_try:
                            samplerates_to_try.append(r)
                    
                    for rate in samplerates_to_try:
                        try:
                            print(f"[AUDIO] Testing device {dev_idx} ({device_name}) @ {rate}Hz...")
                            s = sd.InputStream(
                                device=dev_idx,
                                samplerate=rate,
                                channels=1,
                                callback=callback,
                                blocksize=int(rate * self.config.AUDIO_BUFFER_SECONDS)
                            )
                            s.start()
                            self.current_samplerate = rate
                            print(f"[AUDIO] ✓ Success! Opened stream @ {rate}Hz")
                            return s
                        except Exception as rate_error:
                            # Too much spam if we print every rate error
                            continue
                    return None
                except Exception as e:
                    print(f"[AUDIO] Failed device {dev_idx}: {e}")
                    return None

            # Priority 1: Default device
            try:
                device_info = sd.query_devices(kind='input')
                default_idx = device_info['index']
            except:
                default_idx = None

            if default_idx is not None:
                stream = try_device(default_idx)

            if not stream:
                # Priority 2: Look for WASAPI or MME devices specifically
                devices = sd.query_devices()
                for i, d in enumerate(devices):
                    if d['max_input_channels'] > 0:
                        # Favor WASAPI then MME
                        if wasapi_idx is not None and d['hostapi'] == wasapi_idx:
                            stream = try_device(i)
                            if stream: break
                
                if not stream:
                    for i, d in enumerate(devices):
                        if d['max_input_channels'] > 0 and d['hostapi'] == mme_idx:
                            stream = try_device(i)
                            if stream: break

            if not stream:
                # Fallback: Just any input device
                for i, d in enumerate(sd.query_devices()):
                    if d['max_input_channels'] > 0:
                        stream = try_device(i)
                        if stream: break

            if stream:
                print(f"[AUDIO] ✓ Successfully opened stream on index {stream.device}")

            if stream:
                print(f"[AUDIO] ✓ Successfully opened stream on index {stream.device}")
                while self.running:
                    time.sleep(0.1)
                stream.stop()
                stream.close()
            else:
                print("[AUDIO] CRITICAL: Could not open any audio input device across all Host APIs")
                    
            print("[AUDIO] Audio capture stopped")
        
        except Exception as e:
            print(f"[AUDIO] Error in audio capture: {e}")
    
    def _process_audio(self, audio_data: np.ndarray):
        """Process audio chunk: transcribe and parse commands"""
        
        # Check if there's enough energy (voice activity detection)
        if not self._has_voice_activity(audio_data):
            return
        
        # Transcribe speech
        transcription = self._transcribe(audio_data)
        
        if not transcription or len(transcription.strip()) == 0:
            return
        
        print(f"[AUDIO] Transcribed: '{transcription}'")
        
        self.transcriptions_count += 1
        
        # Update shared memory with latest transcription
        self.shared_memory.update_transcription(transcription)
        
        # Save to transcript file
        self._save_transcript(transcription)
        
        # Parse command
        command = self._parse_command(transcription)
        
        if command:
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_command_time < self.config.COMMAND_COOLDOWN:
                # Still add it but maybe the vision pipeline ignores it if it's too frequent? 
                # Actually, the requirement says "seconds between commands", so we enforce it here.
                return
            
            self.last_command_time = current_time
            self.commands_parsed += 1
            
            # Add to shared memory
            self.shared_memory.add_command(command)
            
            print(f"[AUDIO] ✓ Command parsed: {command.type} - {command.name or command.label}")
    
    def _has_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Simple voice activity detection based on energy"""
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Threshold for silence (can be tuned)
        return energy > 0.005
    
    def _transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to text"""
        try:
            segments, _ = self.speech_model.transcribe(
                audio_data,
                language=self.config.WHISPER_LANGUAGE,
                beam_size=5
            )
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        
        except Exception as e:
            print(f"[AUDIO] Transcription error: {e}")
            return ""
    
    def _parse_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse transcribed text using the NLP Interpreter"""
        # Get structured JSON from interpreter
        result = self.interpreter.interpret(text)
        
        intent = result.get('intent', 'none')
        if intent == 'none':
            return None
            
        # Map interpreter intent back to shared memory VoiceCommand types
        # This matches the user's role mapping:
        # "name_person", "name_object", "self_identification", "query_identity", "query_object"
        
        cmd_type_map = {
            'self_identification': 'assign_self_identity',
            'name_person': 'assign_person_identity',
            'name_object': 'assign_object_label',
            'query_identity': 'query_identity',
            'query_object': 'query_object'
        }
        
        v_type = cmd_type_map.get(intent)
        if not v_type:
            return None
            
        return VoiceCommand(
            type=v_type,
            name=result.get('assigned_name'),
            label=result.get('assigned_name') if intent == 'name_object' else None,
            raw_text=text
        )
    
    def _save_transcript(self, text: str):
        """Save transcription to file"""
        try:
            from datetime import datetime
            import os
            
            os.makedirs(os.path.dirname(self.config.TRANSCRIPT_PATH), exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {text}\n"
            
            with open(self.config.TRANSCRIPT_PATH, 'a', encoding='utf-8') as f:
                f.write(line)
        
        except Exception as e:
            print(f"[AUDIO] Error saving transcript: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'transcriptions': self.transcriptions_count,
            'commands_parsed': self.commands_parsed,
            'queue_size': self.audio_queue.qsize()
        }
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False


# ========== MOCK IMPLEMENTATIONS (Replace with real models) ==========

class MockSpeechRecognizer:
    """Mock speech recognizer - replace with Faster-Whisper"""
    
    def __init__(self, config):
        self.config = config
        
        # In real implementation: load Faster-Whisper
        # from faster_whisper import WhisperModel
        # self.model = WhisperModel(
        #     config.WHISPER_MODEL,
        #     device="cuda" if config.USE_GPU else "cpu",
        #     compute_type=config.WHISPER_COMPUTE_TYPE
        # )
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Mock transcription - returns empty string"""
        
        # In real implementation: transcribe audio
        # segments, info = self.model.transcribe(
        #     audio_data,
        #     language=self.config.WHISPER_LANGUAGE,
        #     beam_size=5
        # )
        # text = " ".join([segment.text for segment in segments])
        # return text
        
        # For demo: return empty (no real audio being captured)
        return ""