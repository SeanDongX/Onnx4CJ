/**
 * test_session.cj — Unit tests for OnnxSession and SessionOptions
 *
 * Tests:
 *   - Loading a non-existent model throws ModelNotFoundException
 *   - SessionOptions can be configured without error
 *   - SessionOptions chain calls return self
 *   - Session input/output count matches expected values (requires model)
 *   - Session input/output names are non-empty strings (requires model)
 *
 * Tests that require a model file are annotated @Ignore when no model
 * is available and can be enabled by setting env var TEST_MODEL_PATH.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.tests

import onnx4cj.core.*
import std.unittest.*
import std.env.*

@TestCase
class SessionOptionsTests {

    /**
     * Default SessionOptions should be created without error.
     */
    @Test
    func testCreateDefaultOptions() {
        let opts = SessionOptions()
        defer { opts.close() }
        // Test passes if creation succeeds without throwing
    }

    /**
     * Chained configuration calls should not throw.
     */
    @Test
    func testConfigureOptions() {
        let opts = SessionOptions()
            .setIntraOpNumThreads(2)
            .setInterOpNumThreads(1)
        defer { opts.close() }
        // Test passes if configuration succeeds without throwing
    }
}

@TestCase
class OnnxSessionTests {

    /**
     * Loading a model from a path that does not exist should throw ModelNotFoundException.
     */
    @Test
    func testMissingModelThrowsException() {
        let env = OnnxEnv("test_session")
        defer { env.close() }
        assertThrows<ModelNotFoundException> {
            let _ = OnnxSession(env, "/nonexistent/path/to/model.onnx")
        }
    }

    /**
     * Loading a model from a path that does not exist (with options) should also throw.
     */
    @Test
    func testMissingModelWithOptionsThrowsException() {
        let env = OnnxEnv("test_session_opts")
        defer { env.close() }
        let opts = SessionOptions()
        defer { opts.close() }
        assertThrows<ModelNotFoundException> {
            let _ = OnnxSession(env, "/nonexistent/model.onnx", Some(opts))
        }
    }

    /**
     * Integration test: load a real model and verify metadata.
     * Requires TEST_MODEL_PATH environment variable to point to an ONNX model file.
     *
     * Run: TEST_MODEL_PATH=src/models/test_add.onnx cjpm test
     */
    @Test
    @ConditionalIgnore(condition: { getEnv("TEST_MODEL_PATH").isEmpty() })
    func testLoadRealModel() {
        let modelPath = getEnv("TEST_MODEL_PATH")
        let env = OnnxEnv("real_model_test")
        defer { env.close() }
        let session = OnnxSession(env, modelPath)
        defer { session.close() }

        // Should have at least one input and one output
        assertTrue(session.inputCount > 0)
        assertTrue(session.outputCount > 0)

        // Input/output names must be non-empty
        for name in session.inputNames {
            assertTrue(!name.isEmpty())
        }
        for name in session.outputNames {
            assertTrue(!name.isEmpty())
        }
    }
}
