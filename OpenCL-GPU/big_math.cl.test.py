#!/usr/bin/env python3

"""
Test suite for big_math.cl - 256-bit arithmetic for Curve25519/Ed25519

BYTE ORDER: all byte<->limb conversions use LITTLE-ENDIAN.
  seed_to_scalar: bytes[0]=LSB -> limb[0]
  scalar_to_bytes: limb[0] -> bytes[0]=LSB
  This matches OpenCL x86/x64 memory layout.
"""

import sys, os, random, argparse
import numpy as np

try:
    import importlib.metadata
    pytools_version = importlib.metadata.version("pytools")
    pytools_tuple = tuple(int(x) for x in pytools_version.split(".")[:3])
    if pytools_tuple < (2025, 1, 12):
        print(f"ERROR: pytools version {pytools_version} is too old")
        sys.exit(1)
    print(f"OK: pytools version {pytools_version} is compatible")
except Exception as e:
    print(f"ERROR: Failed to check pytools version: {e}")
    sys.exit(1)


try:
    import pyopencl as cl
except ImportError as e:
    print(f"Error: pyopencl required - {e}")
    sys.exit(1)

P = 2**255 - 19

def to_256bit_array(value):
    result = np.zeros(8, dtype=np.uint32)
    for i in range(8):
        result[i] = (value >> (32 * i)) & 0xFFFFFFFF
    return result

def from_256bit_array(arr):
    result = 0
    for i in range(8):
        result += int(arr[i]) << (32 * i)
    return result

def get_opencl_device():
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                return device
    raise RuntimeError("No GPU device found")

def load_kernel():
    kernel_path = "./big_math_test_kernels.cl"
    with open(kernel_path, "r") as f:
        return f.read(), kernel_path

def run_opencl_test_single_inout(kernel_code, function_name, input_array):
    device = get_opencl_device()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()
    # Handle both single array and list of arrays
    if isinstance(input_array, list):
        # List format: [(src, dtype), (dst, dtype), ...] or just [src, dst, ...]
        if len(input_array) > 0 and isinstance(input_array[0], tuple):
            # Format: [(src, dtype), (dst, dtype), ...]
            cl_bufs = []
            arrs = []
            for buf, dtype in input_array:
                arr = np.array(buf, dtype=dtype)
                arrs.append(arr)
                cl_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                                  size=len(arr) * arr.dtype.itemsize, hostbuf=arr)
                cl_bufs.append(cl_buf)
            prg.__getattr__(function_name)(queue, (1,), None, *cl_bufs)
        else:
            # Format: [src, dst, ...] - single arrays
            cl_bufs = []
            arrs = []
            for arr in input_array:
                arrs.append(arr)
                cl_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                                  size=len(arr) * arr.dtype.itemsize, hostbuf=arr)
                cl_bufs.append(cl_buf)
            prg.__getattr__(function_name)(queue, (1,), None, *cl_bufs)
        queue.finish()
        # Return the last buffer (the output)
        result = np.empty(len(arrs[-1]), dtype=arrs[-1].dtype)
        cl.enqueue_copy(queue, result, cl_bufs[-1], is_blocking=True)
        return result
    else:
        # Handle tuple (array, dtype)
        if isinstance(input_array, tuple):
            arr = np.array(input_array[0], dtype=input_array[1])
        else:
            arr = input_array.copy()
        cl_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                          size=len(arr) * arr.dtype.itemsize, hostbuf=arr)
        prg.__getattr__(function_name)(queue, (1,), None, cl_buf)
        queue.finish()
        result = np.empty(len(arr), dtype=arr.dtype)
        cl.enqueue_copy(queue, result, cl_buf, is_blocking=True)
        return result

def run_opencl_test_two_out(kernel_code, function_name, input_buffer, output_buffer, output_mode=cl.mem_flags.WRITE_ONLY):
    """For functions that take 2 arguments: input (READ_ONLY) and output (WRITE_ONLY or READ_WRITE)"""
    device = get_opencl_device()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()
    
    # Input buffer
    if isinstance(input_buffer, tuple):
        arr = np.frombuffer(input_buffer[0], dtype=input_buffer[1]) if isinstance(input_buffer[0], bytes) else np.array(input_buffer[0], dtype=input_buffer[1])
    else:
        arr = np.array(input_buffer, dtype=np.uint32)
    cl_input = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                        size=len(arr) * arr.dtype.itemsize, hostbuf=arr)
    
    # Output buffer
    if isinstance(output_buffer, tuple):
        arr_out = np.frombuffer(output_buffer[0], dtype=output_buffer[1]) if isinstance(output_buffer[0], bytes) else np.array(output_buffer[0], dtype=output_buffer[1])
    else:
        arr_out = np.array(output_buffer, dtype=np.uint32)
    # Always initialize output buffer to zeros (needed for | operations in kernels)
    cl_output = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, len(arr_out) * arr_out.dtype.itemsize, hostbuf=arr_out)
    
    prg.__getattr__(function_name)(queue, (1,), None, cl_input, cl_output)
    queue.finish()
    result = np.empty(len(arr_out), dtype=arr_out.dtype)
    cl.enqueue_copy(queue, result, cl_output).wait()
    return result


def run_opencl_test(kernel_code, function_name, input_buffers, output_size, buffer_modes=None, output_mode=cl.mem_flags.WRITE_ONLY):
    """For functions with multiple inputs and one output.
    input_buffers: list of (array, dtype) tuples for inputs
    output_size: size in bytes of the output buffer
    buffer_modes: list of buffer modes (READ_ONLY, WRITE_ONLY, etc.) for each input buffer
    output_mode: mode for the output buffer (default WRITE_ONLY)
    """
    device = get_opencl_device()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()
    cl_bufs = []
    for i, buf in enumerate(input_buffers):
        if isinstance(buf, tuple):
            arr = np.array(buf[0], dtype=buf[1])
        else:
            arr = np.array(buf, dtype=np.uint32)
        # Determine buffer mode
        if buffer_modes and i < len(buffer_modes):
            mode = buffer_modes[i]
        else:
            mode = cl.mem_flags.READ_ONLY
        if mode == cl.mem_flags.WRITE_ONLY:
            # WRITE_ONLY buffer cannot be initialized with hostbuf
            cl_buf = cl.Buffer(ctx, mode, size=len(arr) * arr.dtype.itemsize)
        else:
            # READ_ONLY buffer can be initialized with hostbuf
            cl_buf = cl.Buffer(ctx, mode | cl.mem_flags.COPY_HOST_PTR, 
                             size=len(arr) * arr.dtype.itemsize, hostbuf=arr)
        cl_bufs.append(cl_buf)
    # Determine output buffer mode (default WRITE_ONLY)
    if output_mode == cl.mem_flags.WRITE_ONLY:
        # For WRITE_ONLY, don't initialize with hostbuf
        cl_output = cl.Buffer(ctx, output_mode, output_size)
    else:
        # For other modes, initialize with zeros or None
        cl_output = cl.Buffer(ctx, output_mode, output_size)
    cl_bufs.append(cl_output)  # Add output buffer to the list
    args = cl_bufs
    prg.__getattr__(function_name)(queue, (1,), None, *args)
    queue.finish()
    result = np.empty(output_size // np.dtype(np.uint32).itemsize, dtype=np.uint32)
    cl.enqueue_copy(queue, result, cl_output).wait()
    return result

def numpy_add_256(a, b):
    result = np.zeros(8, dtype=np.uint64)
    carry = 0
    for i in range(8):
        s = int(a[i]) + int(b[i]) + carry
        result[i] = s & 0xFFFFFFFF
        carry = s >> 32
    return result.astype(np.uint32)

def numpy_sub_256(a, b):
    result = np.zeros(8, dtype=np.int64)
    borrow = 0
    for i in range(8):
        d = int(a[i]) - int(b[i]) - borrow
        result[i] = d & 0xFFFFFFFF
        borrow = 1 if d < 0 else 0
    return result.astype(np.uint32)

def numpy_mod_p_reduce(x):
    """Reference: Python big-int modulo P. Ground truth."""
    full_value = sum(int(x[i]) << (32 * i) for i in range(8))
    result_val = full_value % P
    return to_256bit_array(result_val)

def numpy_mul_mod_p(a, b):
    a_val = sum(int(a[i]) << (32*i) for i in range(8))
    b_val = sum(int(b[i]) << (32*i) for i in range(8))
    result_val = (a_val * b_val) % P
    return to_256bit_array(result_val)


def numpy_mod_p_inverse(a):
    a_val = sum(int(a[i]) << (32*i) for i in range(8))
    result_val = pow(a_val, P-2, P)
    return to_256bit_array(result_val)


def numpy_seed_to_scalar(seed):
    result = np.zeros(8, dtype=np.uint32)
    for i in range(32):
        result[i // 4] |= (seed[i] << ((i % 4) * 8))
    return result

def numpy_scalar_to_bytes(scalar):
    """numpy reference: scalar_to_bytes — LITTLE-ENDIAN byte order.
    limb[0] (LSB) -> bytes[0..3], limb[7] (MSB) -> bytes[28..31]"""
    result = bytearray()
    for i in range(8):
        result.extend(bytes([
            (scalar[i] >>  0) & 0xFF,
            (scalar[i] >>  8) & 0xFF,
            (scalar[i] >> 16) & 0xFF,
            (scalar[i] >> 24) & 0xFF,
        ]))
    return bytes(result)

def test_add_256(kernel_code):
    print("Testing add_256...")
    test_cases = [
        ("0 + 0", np.zeros(8, dtype=np.uint32), np.zeros(8, dtype=np.uint32)),
        ("0 + 1", np.zeros(8, dtype=np.uint32), to_256bit_array(1)),
        ("1 + 1", to_256bit_array(1), to_256bit_array(1)),
        ("2^255 + 1", to_256bit_array(2**255), to_256bit_array(1)),
        ("2^256 - 1 + 1", to_256bit_array(2**256 - 1), to_256bit_array(1)),
        ("p+1 + p+1", to_256bit_array(P + 1), to_256bit_array(P + 1)),
    ]
    for i in range(5):
        a_val = random.randint(0, 2**255)
        b_val = random.randint(0, 2**255)
        a_arr = to_256bit_array(a_val)
        b_arr = to_256bit_array(b_val)
        test_cases.append((f"Random pair {i+1}", a_arr, b_arr))
    all_passed = True
    for name, a, b in test_cases:
        expected = numpy_add_256(a, b)
        result = run_opencl_test(kernel_code, "add_256", [(a, np.uint32), (b, np.uint32)], 32, [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
        if np.array_equal(result, expected):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {from_256bit_array(expected)}")
            print(f"    Got:      {from_256bit_array(result)}")
            all_passed = False
    return all_passed

def test_sub_256(kernel_code):
    print("Testing sub_256...")
    # Test cases: zeros, ones, boundary values, and random pairs
    test_cases = [
        # Zero cases
        ("0 - 0", np.zeros(8, dtype=np.uint32), np.zeros(8, dtype=np.uint32)),
        ("1 - 0", to_256bit_array(1), np.zeros(8, dtype=np.uint32)),
        ("0 - 1", np.zeros(8, dtype=np.uint32), to_256bit_array(1)),
        # One cases
        ("1 - 1", to_256bit_array(1), to_256bit_array(1)),
        ("2 - 1", to_256bit_array(2), to_256bit_array(1)),
        # Boundary cases
        ("2^255 - 2^255", to_256bit_array(2**255), to_256bit_array(2**255)),
        ("2^256-1 - 1", to_256bit_array(2**256 - 1), to_256bit_array(1)),
        ("2^256-1 - (2^256-1)", to_256bit_array(2**256 - 1), to_256bit_array(2**256 - 1)),
        # Random pairs
        ("123456789 - 987654321", to_256bit_array(123456789), to_256bit_array(987654321)),
        ("111111111 - 222222222", to_256bit_array(111111111), to_256bit_array(222222222)),
        ("999999999 - 111111111", to_256bit_array(999999999), to_256bit_array(111111111)),
        ("555555555 - 666666666", to_256bit_array(555555555), to_256bit_array(666666666)),
        ("777777777 - 888888888", to_256bit_array(777777777), to_256bit_array(888888888)),
    ]
    all_passed = True
    for name, a, b in test_cases:
        expected = numpy_sub_256(a, b)
        result = run_opencl_test(kernel_code, "sub_256", [(a, np.uint32), (b, np.uint32)], 32, [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
        if np.array_equal(result, expected):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
def test_copy_256(kernel_code):
    print("Testing copy_256...")
    # Test cases: zeros, ones, boundary values, and random values
    test_cases = [
        ("copy zeros", np.zeros(8, dtype=np.uint32)),
        ("copy ones", to_256bit_array(1)),
        ("copy 2^255", to_256bit_array(2**255)),
        ("copy 2^256-1", to_256bit_array(2**256 - 1)),
        ("copy value 1", to_256bit_array(123456789)),
        ("copy value 2", to_256bit_array(987654321)),
        ("copy value 3", to_256bit_array(111111111)),
        ("copy value 4", to_256bit_array(222222222)),
        ("copy value 5", to_256bit_array(333333333)),
        ("copy value 6", to_256bit_array(444444444)),
        ("copy value 7", to_256bit_array(555555555)),
        ("copy value 8", to_256bit_array(666666666)),
        ("copy value 9", to_256bit_array(777777777)),
        ("copy value 10", to_256bit_array(888888888)),
    ]
    all_passed = True
    for name, src in test_cases:
        # For copy_256(src, dst), we pass list of (array, dtype) tuples
        dst = np.zeros(8, dtype=np.uint32)
        result = run_opencl_test_two_out(kernel_code, "copy_256", (src, np.uint32), (dst, np.uint32), cl.mem_flags.READ_WRITE)
        if np.array_equal(result, src):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {src}")
            print(f"    Got:      {dst}")
            all_passed = False
    return all_passed
def test_one_256(kernel_code):
    print("Testing one_256...")
    test_cases = [
        "one_256 test 1",
        "one_256 test 2",
        "one_256 test 3",
        "one_256 test 4",
        "one_256 test 5",
    ]
    all_passed = True
    expected = to_256bit_array(1)
    for name in test_cases:
        result = np.zeros(8, dtype=np.uint32)
        result = run_opencl_test_single_inout(kernel_code, "one_256", (result, np.uint32))
        if np.array_equal(result, expected):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
def test_zero_256(kernel_code):
    print("Testing zero_256...")
    test_cases = [
        "zero_256 test 1",
        "zero_256 test 2",
        "zero_256 test 3",
        "zero_256 test 4",
        "zero_256 test 5",
    ]
    all_passed = True
    expected = np.zeros(8, dtype=np.uint32)
    for name in test_cases:
        result = np.zeros(8, dtype=np.uint32)
        result = run_opencl_test_single_inout(kernel_code, "zero_256", (result, np.uint32))
        if np.array_equal(result, expected):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
def test_mod_p_reduce(kernel_code):
    print("Testing mod_p_reduce...")
    P = 2**255 - 19
    
    test_cases = [
        ("0", to_256bit_array(0)),
        ("1", to_256bit_array(1)),
        ("p-1", to_256bit_array(P - 1)),
        ("p", to_256bit_array(P)),
        ("p+1", to_256bit_array(P + 1)),
        ("p+123", to_256bit_array(P + 123)),
        ("2p", to_256bit_array(2 * P)),
        ("3p", to_256bit_array(3 * P)),
        ("p^2", to_256bit_array(P * P)),
        ("p^2 + 1", to_256bit_array(P * P + 1)),
        ("p^2 + 123", to_256bit_array(P * P + 123)),
        ("Random 1", to_256bit_array(123456789)),
        ("Random 2", to_256bit_array(987654321)),
        ("Random 3", to_256bit_array(111111111)),
        ("Random 4", to_256bit_array(222222222)),
        ("Random 5", to_256bit_array(333333333)),
        ("2^255", to_256bit_array(2**255)),
        ("2^256-1", to_256bit_array(2**256 - 1)),
        ("10*P + 500", to_256bit_array(10 * P + 500)),
    ]
    
    all_passed = True
    for name, input_val in test_cases:
        # Compute expected using Python reference implementation
        expected = numpy_mod_p_reduce(input_val)
        
        result = np.zeros(8, dtype=np.uint32)
        result = run_opencl_test_single_inout(kernel_code, "mod_p_reduce", (input_val, np.uint32))
        
        if np.array_equal(result, expected):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
def test_mul_mod_p(kernel_code):
    print("Testing mul_mod_p...")
    P = 2**255 - 19
    
    test_cases = [
        ("0 * anything", np.zeros(8, dtype=np.uint32), to_256bit_array(12345)),
        ("anything * 0", to_256bit_array(12345), np.zeros(8, dtype=np.uint32)),
        ("1 * 1", to_256bit_array(1), to_256bit_array(1)),
        ("1 * p-1", to_256bit_array(1), to_256bit_array(P - 1)),
        ("2 * 2", to_256bit_array(2), to_256bit_array(2)),
        ("(p-1) * (p-1)", to_256bit_array(P - 1), to_256bit_array(P - 1)),
        ("Random pair 1", to_256bit_array(123456789), to_256bit_array(987654321)),
        ("Random pair 2", to_256bit_array(111111111), to_256bit_array(222222222)),
        ("Random pair 3", to_256bit_array(333333333), to_256bit_array(444444444)),
        ("Random pair 4", to_256bit_array(555555555), to_256bit_array(666666666)),
        ("Random pair 5", to_256bit_array(777777777), to_256bit_array(888888888)),
    ]
    all_passed = True
    for name, a, b in test_cases:
        expected = numpy_mul_mod_p(a, b)
        result = run_opencl_test(kernel_code, "mul_mod_p", [(a, np.uint32), (b, np.uint32)], 32, [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
        if np.array_equal(result, expected):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
def test_mod_p_inverse(kernel_code):
    print("Testing mod_p_inverse...")
    P = 2**255 - 19
    
    test_cases = [
        ("inverse of 1", to_256bit_array(1), to_256bit_array(1)),
        ("inverse of (p-1)", to_256bit_array(P - 1), to_256bit_array(P - 1)),
        ("a * a^(-1) = 1 test 1", to_256bit_array(123456789), to_256bit_array(1)),
        ("a * a^(-1) = 1 test 2", to_256bit_array(987654321), to_256bit_array(1)),
        ("a * a^(-1) = 1 test 3", to_256bit_array(111111111), to_256bit_array(1)),
        ("a * a^(-1) = 1 test 4", to_256bit_array(222222222), to_256bit_array(1)),
        ("a * a^(-1) = 1 test 5", to_256bit_array(333333333), to_256bit_array(1)),
    ]
    all_passed = True
    for name, a, _ in test_cases:
        expected = numpy_mod_p_inverse(a)
        result = run_opencl_test_two_out(kernel_code, "mod_p_inverse", (a, np.uint32), (np.zeros(8, dtype=np.uint32), np.uint32))
        if np.array_equal(result, expected):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    # Additional test: verify a * a^(-1) = 1
    if all_passed:
        for i, (name, a, _) in enumerate(test_cases[2:7], 2):
            inv = run_opencl_test_two_out(kernel_code, "mod_p_inverse", (a, np.uint32), (np.zeros(8, dtype=np.uint32), np.uint32))
            product = run_opencl_test(kernel_code, "mul_mod_p", [(a, np.uint32), (inv, np.uint32)], 32)
            expected_one = to_256bit_array(1)
            if np.array_equal(product, expected_one):
                print(f"  PASS: a * a^(-1) = 1 for test {i}")
            else:
                print(f"  FAIL: a * a^(-1) = 1 for test {i}")
                print(f"    a = {a}")
                print(f"    a^(-1) = {inv}")
                print(f"    a * a^(-1) = {product}")
                all_passed = False
    return all_passed
def test_seed_to_scalar(kernel_code):
    print("Testing seed_to_scalar...")
    test_cases = [
        ("seed 1", bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20])),
        ("seed 2", bytes([0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8, 0xF7, 0xF6, 0xF5, 0xF4, 0xF3, 0xF2, 0xF1, 0xF0, 0xEF, 0xEE, 0xED, 0xEC, 0xEB, 0xEA, 0xE9, 0xE8, 0xE7, 0xE6, 0xE5, 0xE4, 0xE3, 0xE2, 0xE1, 0xE0])),
        ("seed 3", bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01])),
        ("seed 4", bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF])),
        ("seed 5", bytes([0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55])),
        ("seed 6", bytes([0xFF, 0xAA, 0xCC, 0xCC, 0x77, 0x66, 0x44, 0x33, 0x22, 0x55, 0x11, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xCD, 0xFF, 0xFF])),
        ("seed 7", bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01])),
    ]
    all_passed = True
    for name, seed_bytes in test_cases:
        scalar = np.zeros(8, dtype=np.uint32)
        scalar = run_opencl_test_two_out(kernel_code, "seed_to_scalar", (seed_bytes, np.uint8), (scalar, np.uint32))
        expected = np.zeros(8, dtype=np.uint32)
        for i in range(32):
            expected[i//4] |= (seed_bytes[i] << ((i % 4) * 8))
        if np.array_equal(scalar, expected):
            print(f"  PASS: {name} {scalar}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {scalar}")
            all_passed = False
    return all_passed
def test_scalar_to_bytes(kernel_code):
    print("Testing scalar_to_bytes...")
    test_cases = [
        ("scalar 1", to_256bit_array(1)),
        ("scalar 2", to_256bit_array(123456789)),
        ("scalar 3", to_256bit_array(987654321)),
        ("scalar 4", to_256bit_array(111111111)),
        ("scalar 5", to_256bit_array(222222222)),
    ]
    all_passed = True
    for name, scalar in test_cases:
        bytes_out = np.zeros(32, dtype=np.uint8)
        result = run_opencl_test_two_out(kernel_code, "scalar_to_bytes", (scalar, np.uint32), (bytes_out, np.uint8))
        # Expected: LITTLE-ENDIAN byte order (consistent with seed_to_scalar)
        expected = np.zeros(32, dtype=np.uint8)
        for i in range(8):
            expected[i*4 + 0] = (scalar[i] >>  0) & 0xFF
            expected[i*4 + 1] = (scalar[i] >>  8) & 0xFF
            expected[i*4 + 2] = (scalar[i] >> 16) & 0xFF
            expected[i*4 + 3] = (scalar[i] >> 24) & 0xFF
        if np.array_equal(result, expected):
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected.tobytes().hex()}")
            print(f"    Got:      {result.tobytes().hex()}")
            all_passed = False
    return all_passed
def main():
    parser = argparse.ArgumentParser(description="Big Math OpenCL Test Suite")
    parser.add_argument("--func", choices=["all", "add", "sub", "copy", "one", "zero", "mod_p", "mul", "inv", "seed", "scalar"],
                       default="all", help="Test function(s) to run")
    parser.add_argument("--a", type=str, default=None,
                       help="First parameter value (hex or decimal, e.g., 0x123 or 123)")
    parser.add_argument("--b", type=str, default=None,
                       help="Second parameter value (hex or decimal)")
    args = parser.parse_args()
    
    # Parse parameter values
    a_val = 0
    b_val = 0
    if args.a:
        a_val = int(args.a, 0)  # Auto-detect hex (0x) or decimal
    if args.b:
        b_val = int(args.b, 0)
    
    print("=" * 70)
    print("Big Math OpenCL Test Suite")
    print("=" * 70)
    device = get_opencl_device()
    print(f"Device: {device.name}")
    print(f"Platform: {device.platform.name}")
    kernel_code, kernel_path = load_kernel()
    print(f"Kernel: {kernel_path}")
    print()
    
    # Check if custom parameters were provided
    if args.a or args.b:
        # Custom test mode with --a and/or --b parameters
        if args.func == "all":
            print("ERROR: --func all cannot be used with --a/--b parameters")
            sys.exit(1)
        
        if args.func == "add":
            # Test add_256 with custom a and b
            print(f"Testing add_256: {a_val} + {b_val}")
            a_arr = to_256bit_array(a_val)
            b_arr = to_256bit_array(b_val)
            expected = numpy_add_256(a_arr, b_arr)
            result = run_opencl_test(kernel_code, "add_256", [(a_arr, np.uint32), (b_arr, np.uint32)], 32, 
                                    [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(result)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(result)}")
                sys.exit(1)
        elif args.func == "sub":
            # Test sub_256 with custom a and b
            print(f"Testing sub_256: {a_val} - {b_val}")
            a_arr = to_256bit_array(a_val)
            b_arr = to_256bit_array(b_val)
            expected = numpy_sub_256(a_arr, b_arr)
            result = run_opencl_test(kernel_code, "sub_256", [(a_arr, np.uint32), (b_arr, np.uint32)], 32, 
                                    [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(result)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(result)}")
                sys.exit(1)
        elif args.func == "copy":
            # Test copy_256 with custom a
            print(f"Testing copy_256: copy {a_val}")
            a_arr = to_256bit_array(a_val)
            dst = np.zeros(8, dtype=np.uint32)
            expected = a_arr.copy()
            result = run_opencl_test_two_out(kernel_code, "copy_256", (a_arr, np.uint32), (dst, np.uint32), cl.mem_flags.READ_WRITE)
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(result)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(result)}")
                sys.exit(1)
        elif args.func == "one":
            # Test one_256 (no parameters needed)
            print("Testing one_256")
            result = np.zeros(8, dtype=np.uint32)
            expected = to_256bit_array(1)
            result = run_opencl_test_single_inout(kernel_code, "one_256", (result, np.uint32))
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(result)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(result)}")
                sys.exit(1)
        elif args.func == "zero":
            # Test zero_256 (no parameters needed)
            print("Testing zero_256")
            result = np.zeros(8, dtype=np.uint32)
            expected = np.zeros(8, dtype=np.uint32)
            result = run_opencl_test_single_inout(kernel_code, "zero_256", (result, np.uint32))
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(result)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(result)}")
                sys.exit(1)
        elif args.func == "mod_p":
            # Test mod_p_reduce with custom a
            print(f"Testing mod_p_reduce: {a_val} mod (2^255 - 19)")
            a_arr = to_256bit_array(a_val)
            expected = numpy_mod_p_reduce(a_arr)
            result = np.zeros(8, dtype=np.uint32)
            result = run_opencl_test_single_inout(kernel_code, "mod_p_reduce", (a_arr, np.uint32))
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(result)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(result)}")
                sys.exit(1)
        elif args.func == "mul":
            # Test mul_mod_p with custom a and b
            print(f"Testing mul_mod_p: ({a_val} * {b_val}) mod (2^255 - 19)")
            a_arr = to_256bit_array(a_val)
            b_arr = to_256bit_array(b_val)
            expected = numpy_mul_mod_p(a_arr, b_arr)
            result = run_opencl_test(kernel_code, "mul_mod_p", [(a_arr, np.uint32), (b_arr, np.uint32)], 32, 
                                    [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(result)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(result)}")
                sys.exit(1)
        elif args.func == "inv":
            # Test mod_p_inverse with custom a
            print(f"Testing mod_p_inverse: {a_val}^(-1) mod (2^255 - 19)")
            a_arr = to_256bit_array(a_val)
            expected = numpy_mod_p_inverse(a_arr)
            result = np.zeros(8, dtype=np.uint32)
            result = run_opencl_test_two_out(kernel_code, "mod_p_inverse", (a_arr, np.uint32), (result, np.uint32))
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(result)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(result)}")
                sys.exit(1)
        elif args.func == "seed":
            # Test seed_to_scalar with custom seed
            print("Testing seed_to_scalar")
            if a_val:
                # Convert a_val to seed bytes (treat as number of bytes or use default)
                seed_bytes = bytes([i % 256 for i in range(32)])
            else:
                seed_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
            scalar = np.zeros(8, dtype=np.uint32)
            expected = numpy_seed_to_scalar(seed_bytes)
            scalar = run_opencl_test_two_out(kernel_code, "seed_to_scalar", (seed_bytes, np.uint8), (scalar, np.uint32))
            if np.array_equal(scalar, expected):
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(scalar)}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(scalar)}")
                sys.exit(1)
        elif args.func == "scalar":
            # Test scalar_to_bytes with custom a
            print(f"Testing scalar_to_bytes: {a_val}")
            a_arr = to_256bit_array(a_val)
            bytes_out = np.zeros(32, dtype=np.uint8)
            expected = numpy_scalar_to_bytes(a_arr)
            result = run_opencl_test_two_out(kernel_code, "scalar_to_bytes", (a_arr, np.uint32), (bytes_out, np.uint8))
            if np.array_equal(result, expected):
                print(f"  PASS")
                print(f"    Result: {result.tobytes().hex()}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {expected.hex()}")
                print(f"    Got:      {result.tobytes().hex()}")
                sys.exit(1)
    else:
        # Original test mode - run all tests as before
        all_passed = True
        if args.func == "all" or args.func == "add":
            if not test_add_256(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "sub":
            if not test_sub_256(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "copy":
            if not test_copy_256(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "one":
            if not test_one_256(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "zero":
            if not test_zero_256(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "mod_p":
            if not test_mod_p_reduce(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "mul":
            if not test_mul_mod_p(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "inv":
            if not test_mod_p_inverse(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "seed":
            if not test_seed_to_scalar(kernel_code):
                all_passed = False
            print()
        if args.func == "all" or args.func == "scalar":
            if not test_scalar_to_bytes(kernel_code):
                all_passed = False
            print()
        print("=" * 70)
        if all_passed:
            print("RESULT: ALL TESTS PASSED")
        else:
            print("RESULT: SOME TESTS FAILED")
        print("=" * 70)
        return 0 if all_passed else 1
    
    print("=" * 70)
    print("RESULT: TEST PASSED")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
