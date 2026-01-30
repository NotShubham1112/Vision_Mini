"""Root redirect for the vision system."""
import sys
import os

# Add the current directory to sys.path to ensure 'src' is findable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.core.main import main
except ImportError as e:
    print(f"Error: Could not find project structure. {e}")
    sys.exit(1)

if __name__ == "__main__":
    main()
