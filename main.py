"""Main entry point for the multimodal vision system."""

import threading
import signal
import sys
import time
import cv2
from config import SystemConfig
from shared_memory import SharedMemorySystem
from vision_pipeline import VisionPipeline

def main() -> None:
    """Bootstrap the vision system."""
    print("\n" + "="*60)
    print("          VISION REASONING SYSTEM STARTING")
    print("="*60 + "\n")

    # 1. Initialize configuration and shared memory
    config = SystemConfig()
    shared_memory = SharedMemorySystem()

    # 2. Initialize pipelines
    vision = VisionPipeline(config, shared_memory)

    try:
        # 3. Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\n[SYSTEM] Shutdown initiated...")
            vision.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # 4. Initialize models
        print("[SYSTEM] Initializing models...")
        vision._initialize_models()

        # 5. Start vision pipeline in the main thread
        print("[SYSTEM] Starting vision pipeline...")
        vision.run()
    except Exception as e:
        print(f"\n[CRITICAL] System failure: {e}")
        import traceback
        traceback.print_exc()
    finally:
        vision.stop()
        print("[SYSTEM] System stopped. Goodbye!")

if __name__ == "__main__":
    main()
