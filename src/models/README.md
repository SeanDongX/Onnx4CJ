# ONNX Model Files

This directory contains ONNX model files used for testing and examples.

## Test Models (auto-generated)

Generate the test models used by unit tests:

```bash
pip install onnx numpy
python3 scripts/generate_test_models.py
```

This creates:
- `test_add.onnx` — float32 addition: C = A + B
- `test_identity.onnx` — identity: Y = X

## Demo Models (downloaded)

Download pre-trained models for the end-to-end demo:

```bash
python3 scripts/download_models.py resnet50
```

## Note

Large model files (*.onnx, *.pb) are excluded from git via `.gitignore`.
Only the small test models can be committed (< 1 MB).
