/**
 * test_env.cj — Unit tests for OnnxEnv
 *
 * Tests:
 *   - Successful creation with a valid log ID
 *   - Successful creation with empty log ID
 *   - Multiple OnnxEnv instances can be created independently
 *   - OnnxEnv.close() does not panic
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.tests

import onnx4cj.core.*
import std.unittest.*

@TestCase
class OnnxEnvTests {

    /**
     * Creating an OnnxEnv with a valid log ID should succeed.
     * (Test passes if no exception is thrown.)
     */
    @Test
    func testCreateWithLogId() {
        let env = OnnxEnv("test_env_log_id")
        defer { env.close() }
    }

    /**
     * Creating an OnnxEnv with an empty string log ID should succeed.
     */
    @Test
    func testCreateWithEmptyLogId() {
        let env = OnnxEnv("")
        defer { env.close() }
    }

    /**
     * Multiple independent OnnxEnv instances should coexist.
     */
    @Test
    func testMultipleEnvInstances() {
        let env1 = OnnxEnv("env_1")
        let env2 = OnnxEnv("env_2")
        defer {
            env1.close()
            env2.close()
        }
        // Test passes if both envs were created and released without error
    }

    /**
     * OnnxEnv used in a `use` block should close automatically without panicking.
     */
    @Test
    func testUseBlock() {
        use env = OnnxEnv("env_use_block") {
            // Test passes if scope exit is clean
        }
    }
}
