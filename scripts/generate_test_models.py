#!/usr/bin/env python3
"""
generate_test_models.py — Generate minimal ONNX models for unit testing.

Generated models:
  - src/models/test_add.onnx     : y = a + b (float32, shape [1])
  - src/models/test_identity.onnx: y = x (float32, any shape)

Usage:
  pip install onnx numpy
  python3 scripts/generate_test_models.py
"""

import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from onnx.checker import check_model
except ImportError:
    print("ERROR: onnx package not found. Install with: pip install onnx numpy")
    raise SystemExit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_add_model():
    """Create a model that computes C = A + B for float32 scalars."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1])

    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([add_node], "add_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    check_model(model)

    path = os.path.join(OUTPUT_DIR, "test_add.onnx")
    onnx.save(model, path)
    print(f"Created: {path}")


def make_identity_model():
    """Create a model that returns its input unchanged."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, None)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    id_node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([id_node], "identity_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    check_model(model)

    path = os.path.join(OUTPUT_DIR, "test_identity.onnx")
    onnx.save(model, path)
    print(f"Created: {path}")


if __name__ == "__main__":
    make_add_model()
    make_identity_model()
    print("Done. Test models written to src/models/")
