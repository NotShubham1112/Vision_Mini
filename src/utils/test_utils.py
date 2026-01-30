#!/usr/bin/env python3
"""
Testing and Debugging Utilities
Tools for testing individual components and debugging issues
"""

import cv2
import numpy as np
import time
import sys


def test_camera():
    """Test camera capture"""
    print("\n" + "=" * 60)
    print("CAMERA TEST")
    print("=" * 60)
    print("\nTesting camera access...")
    print("Press 'q' to quit, 's' to save a frame\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return False
    
    print("✓ Camera opened successfully")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            
            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"test_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved frame to {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Camera test complete. Average FPS: {frame_count / (time.time() - start_time):.1f}")
    return True


def test_microphone():
    """Test microphone capture"""
    print("\n" + "=" * 60)
    print("MICROPHONE TEST")
    print("=" * 60)
    
    try:
        import sounddevice as sd
        
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        
        print("\nRecording 3 seconds of audio...")
        
        duration = 3
        sample_rate = 16000
        
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float32')
        
        print("Recording... (speak now)")
        sd.wait()
        
        print("✓ Recording complete")
        
        # Calculate audio statistics
        rms = np.sqrt(np.mean(audio ** 2))
        max_amp = np.max(np.abs(audio))
        
        print(f"\nAudio statistics:")
        print(f"  RMS: {rms:.4f}")
        print(f"  Max amplitude: {max_amp:.4f}")
        
        if max_amp < 0.01:
            print("\n⚠ Very low audio level detected")
            print("   Make sure microphone is not muted and volume is up")
        else:
            print("\n✓ Microphone is working")
        
        # Play back
        response = input("\nPlay back recording? (y/n): ").strip().lower()
        if response == 'y':
            print("Playing...")
            sd.play(audio, sample_rate)
            sd.wait()
            print("✓ Playback complete")
        
        return True
    
    except ImportError:
        print("❌ sounddevice not installed")
        print("   Run: pip install sounddevice")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_speech_recognition():
    """Test speech recognition"""
    print("\n" + "=" * 60)
    print("SPEECH RECOGNITION TEST")
    print("=" * 60)
    
    try:
        from faster_whisper import WhisperModel
        import sounddevice as sd
        
        print("\nLoading Whisper model (this may take a minute)...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✓ Model loaded")
        
        print("\nRecording 5 seconds...")
        duration = 5
        sample_rate = 16000
        
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float32')
        
        print("Recording... (speak a command)")
        sd.wait()
        
        print("✓ Recording complete")
        print("Transcribing...")
        
        # Transcribe
        segments, info = model.transcribe(audio.flatten(), beam_size=5)
        
        text = " ".join([segment.text for segment in segments])
        
        if text.strip():
            print(f"\n✓ Transcription: '{text}'")
            print(f"   Language: {info.language}")
            print(f"   Probability: {info.language_probability:.2f}")
        else:
            print("\n⚠ No speech detected")
        
        return True
    
    except ImportError as e:
        print(f"❌ Required package not installed: {e}")
        print("   Run: pip install faster-whisper sounddevice")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_object_detection():
    """Test object detection"""
    print("\n" + "=" * 60)
    print("OBJECT DETECTION TEST")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        print("\nLoading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        print("✓ Model loaded")
        
        print("\nOpening camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        print("✓ Camera opened")
        print("\nRunning detection... (press 'q' to quit)")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Run detection
            results = model(frame, verbose=False)
            
            # Draw results
            annotated = results[0].plot()
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Object Detection Test', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Detection test complete. Average FPS: {frame_count / (time.time() - start_time):.1f}")
        return True
    
    except ImportError:
        print("❌ ultralytics not installed")
        print("   Run: pip install ultralytics")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_face_recognition():
    """Test face recognition"""
    print("\n" + "=" * 60)
    print("FACE RECOGNITION TEST")
    print("=" * 60)
    
    try:
        from insightface.app import FaceAnalysis
        
        print("\nLoading InsightFace model...")
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0)
        print("✓ Model loaded")
        
        print("\nOpening camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        print("✓ Camera opened")
        print("\nDetecting faces... (press 'q' to quit)")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect faces
            faces = app.get(frame)
            
            # Draw faces
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # Draw landmarks
                if hasattr(face, 'kps'):
                    for kp in face.kps:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)
            
            # Show count
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✓ Face recognition test complete")
        return True
    
    except ImportError:
        print("❌ insightface not installed")
        print("   Run: pip install insightface onnxruntime-gpu")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test menu"""
    print("\n" + "🧪 " * 20)
    print(" " * 15 + "TESTING & DEBUG UTILITIES")
    print("🧪 " * 20)
    
    tests = [
        ("Camera", test_camera),
        ("Microphone", test_microphone),
        ("Speech Recognition", test_speech_recognition),
        ("Object Detection (YOLOv8)", test_object_detection),
        ("Face Recognition (InsightFace)", test_face_recognition),
    ]
    
    while True:
        print("\n" + "=" * 60)
        print("Select a test:")
        print("=" * 60)
        
        for i, (name, _) in enumerate(tests, 1):
            print(f"{i}. {name}")
        
        print("0. Exit")
        
        try:
            choice = input("\nEnter choice: ").strip()
            
            if choice == '0':
                print("\nExiting...")
                break
            
            choice = int(choice)
            
            if 1 <= choice <= len(tests):
                name, test_func = tests[choice - 1]
                test_func()
            else:
                print("Invalid choice")
        
        except ValueError:
            print("Invalid input")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


if __name__ == "__main__":
    main()