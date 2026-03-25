/**
 * ort_types.cj — ONNX Runtime type definitions for Cangjie FFI
 *
 * Mirrors the opaque handle types and enum values defined in ort_shim.h.
 * These are used throughout the ffi and core layers.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.ffi

import std.core.*

// ---------------------------------------------------------------------------
// Opaque C handle aliases
// ---------------------------------------------------------------------------

// Raw pointer width — on 64-bit platforms this is Int64
public typealias OrtEnvHandle         = CPointer<Unit>
public typealias OrtSessionHandle     = CPointer<Unit>
public typealias OrtSessionOptsHandle = CPointer<Unit>
public typealias OrtValueHandle       = CPointer<Unit>

// ---------------------------------------------------------------------------
// Status codes (must match ORT_SHIM_STATUS in ort_shim.h)
// ---------------------------------------------------------------------------

public enum OrtShimStatus <: Equatable<OrtShimStatus> {
    | Ok             // 0
    | Fail           // 1
    | InvalidArgument// 2
    | NoSuchFile     // 3
    | NoModel        // 4
    | EngineError    // 5
    | RuntimeException// 6
    | InvalidProtobuf// 7
    | ModelLoaded    // 8
    | NotImplemented // 9
    | InvalidGraph   // 10
    | EpFail         // 11

    public static func fromInt(code: Int32): OrtShimStatus {
        match code {
            case 0  => Ok
            case 1  => Fail
            case 2  => InvalidArgument
            case 3  => NoSuchFile
            case 4  => NoModel
            case 5  => EngineError
            case 6  => RuntimeException
            case 7  => InvalidProtobuf
            case 8  => ModelLoaded
            case 9  => NotImplemented
            case 10 => InvalidGraph
            case _  => EpFail
        }
    }

    public func toInt(): Int32 {
        match this {
            case Ok              => 0
            case Fail            => 1
            case InvalidArgument => 2
            case NoSuchFile      => 3
            case NoModel         => 4
            case EngineError     => 5
            case RuntimeException=> 6
            case InvalidProtobuf => 7
            case ModelLoaded     => 8
            case NotImplemented  => 9
            case InvalidGraph    => 10
            case EpFail          => 11
        }
    }

    public operator func ==(other: OrtShimStatus): Bool {
        this.toInt() == other.toInt()
    }

    public operator func !=(other: OrtShimStatus): Bool {
        !(this == other)
    }
}

// ---------------------------------------------------------------------------
// Element types (must match ORT_ELEMENT_TYPE in ort_shim.h)
// ---------------------------------------------------------------------------

public enum OrtElementType <: Equatable<OrtElementType> {
    | Undefined    // 0
    | Float32      // 1
    | UInt8        // 2
    | Int8         // 3
    | UInt16       // 4
    | Int16        // 5
    | Int32        // 6
    | Int64        // 7
    | String       // 8
    | Bool         // 9
    | Float16      // 10
    | Float64      // 11
    | UInt32       // 12
    | UInt64       // 13
    | Complex64    // 14
    | Complex128   // 15
    | BFloat16     // 16

    public static func fromInt(code: Int32): OrtElementType {
        match code {
            case 1  => Float32
            case 2  => UInt8
            case 3  => Int8
            case 4  => UInt16
            case 5  => Int16
            case 6  => Int32
            case 7  => Int64
            case 8  => String
            case 9  => Bool
            case 10 => Float16
            case 11 => Float64
            case 12 => UInt32
            case 13 => UInt64
            case 14 => Complex64
            case 15 => Complex128
            case 16 => BFloat16
            case _  => Undefined
        }
    }

    public func toInt(): Int32 {
        match this {
            case Undefined  => 0
            case Float32    => 1
            case UInt8      => 2
            case Int8       => 3
            case UInt16     => 4
            case Int16      => 5
            case Int32      => 6
            case Int64      => 7
            case String     => 8
            case Bool       => 9
            case Float16    => 10
            case Float64    => 11
            case UInt32     => 12
            case UInt64     => 13
            case Complex64  => 14
            case Complex128 => 15
            case BFloat16   => 16
        }
    }

    public operator func ==(other: OrtElementType): Bool {
        this.toInt() == other.toInt()
    }

    public operator func !=(other: OrtElementType): Bool {
        !(this == other)
    }
}
