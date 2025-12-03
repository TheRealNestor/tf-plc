"""
PLC type system utilities.
"""

import numpy as np
from typing import Tuple



def plc_type_from_onnx_dtype(dtype: str) -> str:
    """Map ONNX data type strings to PLC data types."""
    if dtype is None:
        raise ValueError("IR layer data type is None; tensor_info is incomplete.")

    match dtype:
        case "TensorProto.FLOAT":
            return "REAL"
        case "TensorProto.DOUBLE":
            return "LREAL"
        case "TensorProto.INT32":
            return "DINT"
        case "TensorProto.INT64":
            return "LINT"
        case "TensorProto.INT8":
            return "SINT"
        case "TensorProto.UINT8":
            return "USINT"
        case "TensorProto.INT16":
            return "INT"
        case "TensorProto.UINT16":
            return "UINT"
        case _:
            raise NotImplementedError(f"Data type {dtype} is not supported.")


def numpy_to_plc_type(dtype: np.dtype) -> str:
    """Convert numpy dtype to IEC 61131-3 type."""
    type_map = {
        np.int8: "SINT",
        np.uint8: "USINT",
        np.int16: "INT",
        np.uint16: "UINT",
        np.int32: "DINT",
        np.uint32: "UDINT",
        np.int64: "LINT",
        np.float32: "REAL",
        np.float64: "LREAL",
    }

    if dtype.type not in type_map:
        raise NotImplementedError(f"Numpy dtype {dtype} is not supported.")
    
    return type_map[dtype.type]

def numpy_to_plc_cast_func(np_dtype: np.dtype, target_plc_type: str) -> str:
    """Get PLC cast function from numpy dtype to target type."""
    if target_plc_type == "REAL":
        if np_dtype == np.int8:
            return "SINT_TO_REAL"
        elif np_dtype == np.uint8:
            return "USINT_TO_REAL"
        elif np_dtype == np.int16:
            return "INT_TO_REAL"
        elif np_dtype == np.uint16:
            return "UINT_TO_REAL"
        elif np_dtype == np.int32:
            return "DINT_TO_REAL"
        elif np_dtype == np.uint32:
            return "UDINT_TO_REAL"
        else:
            raise ValueError(f"Unsupported conversion from {np_dtype} to {target_plc_type}")
    else:
        raise ValueError(f"Unsupported target type: {target_plc_type}")

def get_accumulator_type(dtype: np.dtype) -> str:
    """Get appropriate accumulator type for quantized operations."""
    if dtype in [np.int8, np.uint8]:
        return "DINT"  # 32-bit accumulator for 8-bit
    elif dtype in [np.int16, np.uint16]:
        return "LINT"  # 64-bit accumulator for 16-bit
    else:
        return "REAL"


def get_type_limits(dtype: np.dtype) -> Tuple[int, int]:
    """Get min/max values for a dtype."""
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    if info:
        return (info.min, info.max)
    return (0, 0)  # For floats


def get_conversion_func(from_type: str, to_type: str) -> str:
    """Get PLC type conversion function name."""
    if from_type == to_type:
        return ""  # No conversion needed
    return f"{from_type}_TO_{to_type}"
