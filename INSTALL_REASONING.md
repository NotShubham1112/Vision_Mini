# Vision-Language Reasoning System - Installation & Setup

## 📦 Installation Steps

### Prerequisites
- Windows 10/11
- Python 3.8-3.11
- NVIDIA GPU with CUDA support (RTX 3050 or better)
- ~15GB free disk space (for models)
- ~8GB VRAM

---

### Step 1: Install Core Dependencies

```bash
cd d:\Vision_aI\multimodal_vision_system

# Activate virtual environment
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 2: Install Vision-Language Dependencies

```bash
# CLIP and image processing
pip install open-clip-torch ftfy regex Pillow

# Transformers for LLaVA
pip install transformers accelerate sentencepiece protobuf

# 4-bit quantization (for LLaVA)
pip install bitsandbytes

# FAISS for similarity search
pip install faiss-gpu
# If faiss-gpu fails, use: pip install faiss-cpu
```

---

### Step 3: Verify Installation

```python
# Test imports
python -c "import open_clip; print('✓ CLIP installed')"
python -c "import faiss; print('✓ FAISS installed')"
python -c "import transformers; print('✓ Transformers installed')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

### Step 4: Build Embedding Database

```bash
# Build sample database (quick start)
python build_embedding_database.py --mode sample --num-classes 20

# Expected output:
# ✓ Sample database created!
#   Location: data/embeddings
```

---

### Step 5: Test Components

```bash
# Test CLIP only (fast)
python test_reasoning.py --test clip

# Test fusion logic
python test_reasoning.py --test fusion

# Full test without LLaVA
python test_reasoning.py --skip-llava
```

---

## 🚀 Running the System

### Basic Usage

```bash
python main.py
```

### Configuration Options

Edit `config.py` before running:

```python
# Quick performance modes:

# MODE 1: Full Reasoning (Slow, Most Accurate)
USE_VLM_REASONING = True
USE_EMBEDDING_DB = True
YOLO_MODEL = "yolov8m.pt"

# MODE 2: CLIP Only (Fast, Good Accuracy)
USE_VLM_REASONING = False
USE_EMBEDDING_DB = True
YOLO_MODEL = "yolov8m.pt"

# MODE 3: YOLO Only (Fastest, Basic)
USE_VLM_REASONING = False
USE_EMBEDDING_DB = False
YOLO_MODEL = "yolov8n.pt"
```

---

## 🔧 Troubleshooting

### NumPy Warning
If you see "Numpy built with MINGW-W64" warning:
```bash
pip uninstall numpy
pip install numpy==1.24.3
```

### CUDA Out of Memory
```python
# In config.py:
USE_VLM_REASONING = False  # Disable LLaVA
YOLO_MODEL = "yolov8n.pt"  # Use smaller YOLO
```

### FAISS GPU Issues
```bash
pip uninstall faiss-gpu
pip install faiss-cpu
```

### Slow First Run
- LLaVA downloads ~4GB on first use
- CLIP downloads ~400MB on first use
- Be patient, subsequent runs are fast

---

## 📊 Expected Performance

### RTX 3050 (8GB VRAM)

| Mode | FPS | VRAM | Accuracy |
|------|-----|------|----------|
| YOLO Only | 25-30 | ~3GB | Good |
| YOLO + CLIP | 20-25 | ~4GB | Better |
| Full Reasoning | 15-20 | ~8GB | Best |

### Reasoning Overhead

- CLIP embedding: ~5ms per object
- FAISS search: <1ms per query
- LLaVA reasoning: ~500ms per object (only for uncertain cases)

---

## ✅ Verification Checklist

Before using in production:

- [ ] All dependencies installed without errors
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] Embedding database built successfully
- [ ] Test suite passes (at least CLIP and fusion tests)
- [ ] System runs without crashes
- [ ] GPU memory usage acceptable
- [ ] FPS acceptable for your use case

---

## 📚 Next Steps

1. **Test with real objects** - See how it performs
2. **Tune thresholds** - Adjust fusion confidence levels
3. **Build custom database** - Add your own object images
4. **Benchmark performance** - Measure FPS and accuracy

---

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review [QUICKSTART_REASONING.md](file:///d:/Vision_aI/multimodal_vision_system/QUICKSTART_REASONING.md)
3. Check [walkthrough.md](file:///C:/Users/LOQ/.gemini/antigravity/brain/a7958321-b11e-41de-ba83-49ef3122b2b6/walkthrough.md) for details
4. Verify GPU drivers are up to date

---

**Installation complete!** Proceed to [QUICKSTART_REASONING.md](file:///d:/Vision_aI/multimodal_vision_system/QUICKSTART_REASONING.md) for usage instructions.
