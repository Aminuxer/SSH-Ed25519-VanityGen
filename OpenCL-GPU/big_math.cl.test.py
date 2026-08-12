#!/usr/bin/env python3

"""
Test suite for big_math.cl - 256-bit arithmetic for Curve25519/Ed25519

BYTE ORDER: all byte<->limb conversions use LITTLE-ENDIAN.
  seed_to_scalar: bytes[0]=LSB -> limb[0]
  scalar_to_bytes: limb[0] -> bytes[0]=LSB
  This matches OpenCL x86/x64 memory layout.
"""

import sys, os, random, argparse
import array
import ctypes

try:
    import pyopencl as cl
except ImportError as e:
    print(f"Error: pyopencl required - {e}")
    sys.exit(1)

P = 2**255 - 19

# ctypes helpers (bigint arrays)
c_uint64_4 = ctypes.c_uint64 * 4
c_uint8_32 = ctypes.c_uint8 * 32

def py_to_ctypes_ints(lst, n):
    """Convert Python list of ints to ctypes array of c_uint64."""
    arr = c_uint64_4(*[ctypes.c_uint64(v) for v in lst])
    return arr

def py_to_ctypes_bytes(lst):
    """Convert Python list of ints to ctypes array of c_uint8 (32 bytes)."""
    arr = c_uint8_32(*[ctypes.c_uint8(v) for v in lst])
    return arr

def ctypes_to_list(ct_arr):
    """Convert ctypes array to Python list of ints."""
    return [int(v) for v in ct_arr]

def bytes_from_ctypes(ct_arr):
    """Convert ctypes c_uint8 array to bytes."""
    return bytes(ct_arr)

"""Convert integer to 4-element list of ints (little-endian limbs)."""
def to_256bit_array(value):
    result = [0, 0, 0, 0]
    for i in range(4):
        result[i] = (value >> (64 * i)) & 0xFFFFFFFFFFFFFFFF
    return result

"""Convert 4-element list of ints to integer (little-endian limbs)."""
def from_256bit_array(arr):
    result = 0
    for i in range(4):
        result += int(arr[i]) << (64 * i)
    return result

"""Find and return the first available GPU OpenCL device."""
def get_opencl_device():
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                return device
    raise RuntimeError("No GPU device found")

"""Load and return the OpenCL kernel source code from big_math_test_kernels.cl."""
def load_kernel():
    kernel_path = "./big_math_test_kernels.cl"
    with open(kernel_path, "r") as f:
        return f.read(), kernel_path

def _buf_size(arr, dtype):
    """Compute buffer size in bytes for a Python list/array."""
    itemsize = 8 if dtype == ctypes.c_uint64 else 1
    return len(arr) * itemsize

"""Run a single-input/single-output OpenCL test kernel, return output list."""
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
                if dtype == ctypes.c_uint64:
                    ct_arr = py_to_ctypes_ints(buf, 4)
                else:
                    ct_arr = py_to_ctypes_bytes(buf)
                arrs.append(ct_arr)
                cl_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                  size=_buf_size(buf, dtype), hostbuf=ct_arr)
                cl_bufs.append(cl_buf)
            prg.__getattr__(function_name)(queue, (1,), None, *cl_bufs)
        else:
            # Format: [src, dst, ...] - single arrays
            cl_bufs = []
            arrs = []
            for arr in input_array:
                arrs.append(arr)
                ct_arr = py_to_ctypes_ints(arr, len(arr))
                cl_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                  size=len(arr) * 8, hostbuf=ct_arr)
                cl_bufs.append(cl_buf)
            prg.__getattr__(function_name)(queue, (1,), None, *cl_bufs)
        queue.finish()
        # Read back the last buffer into a Python list
        result_arr = c_uint64_4()
        cl.enqueue_copy(queue, result_arr, cl_bufs[-1], is_blocking=True)
        return ctypes_to_list(result_arr)
    else:
        # Handle tuple (array, dtype)
        if isinstance(input_array, tuple):
            lst, dtype = input_array[0], input_array[1]
        else:
            lst = list(input_array)
            dtype = ctypes.c_uint64
        if dtype == ctypes.c_uint64:
            ct_arr = py_to_ctypes_ints(lst, 4)
        else:
            ct_arr = py_to_ctypes_bytes(lst)
        cl_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                          size=len(lst) * (8 if dtype == ctypes.c_uint64 else 1), hostbuf=ct_arr)
        prg.__getattr__(function_name)(queue, (1,), None, cl_buf)
        queue.finish()
        result_arr = c_uint64_4()
        cl.enqueue_copy(queue, result_arr, cl_buf, is_blocking=True)
        return ctypes_to_list(result_arr)

"""Run an OpenCL test kernel with one input and one output buffer."""
def run_opencl_test_two_out(kernel_code, function_name, input_buffer, output_buffer, output_mode=cl.mem_flags.WRITE_ONLY):
    """For functions that take 2 arguments: input (READ_ONLY) and output (WRITE_ONLY or READ_WRITE)"""
    device = get_opencl_device()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()

    # Input buffer
    if isinstance(input_buffer, tuple):
        inp_data, inp_dtype = input_buffer[0], input_buffer[1]
    else:
        inp_data, inp_dtype = input_buffer, ctypes.c_uint64
    if inp_dtype == ctypes.c_uint64:
        inp_ct = py_to_ctypes_ints(inp_data, 4)
        inp_size = len(inp_data) * 8
    else:
        inp_ct = py_to_ctypes_bytes(inp_data)
        inp_size = len(inp_data)
    cl_input = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, inp_size, hostbuf=inp_ct)

    # Output buffer
    if isinstance(output_buffer, tuple):
        out_data, out_dtype = output_buffer[0], output_buffer[1]
    else:
        out_data, out_dtype = output_buffer, ctypes.c_uint64
    if out_dtype == ctypes.c_uint64:
        out_size = len(out_data) * 8
    else:
        out_size = len(out_data)

    if output_mode == cl.mem_flags.WRITE_ONLY:
        cl_output = cl.Buffer(ctx, output_mode, out_size)
    elif output_mode == cl.mem_flags.READ_WRITE:
        out_ct = py_to_ctypes_ints(out_data, 4) if out_dtype == ctypes.c_uint64 else py_to_ctypes_bytes(out_data)
        cl_output = cl.Buffer(ctx, output_mode | cl.mem_flags.COPY_HOST_PTR, out_size, hostbuf=out_ct)
    else:
        cl_output = cl.Buffer(ctx, output_mode, out_size)

    prg.__getattr__(function_name)(queue, (1,), None, cl_input, cl_output)
    queue.finish()

    # Read back result
    if out_dtype == ctypes.c_uint64:
        result_arr = c_uint64_4()
    else:
        result_arr = c_uint8_32()
    cl.enqueue_copy(queue, result_arr, cl_output).wait()

    if out_dtype == ctypes.c_uint64:
        return ctypes_to_list(result_arr)
    else:
        return ctypes_to_list(result_arr)


"""Run an OpenCL test kernel with multiple input buffers and one output buffer."""
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
            data, dtype = buf[0], buf[1]
        else:
            data, dtype = buf, ctypes.c_uint64
        if dtype == ctypes.c_uint64:
            ct_arr = py_to_ctypes_ints(data, 4)
            bufsz = len(data) * 8
        else:
            ct_arr = py_to_ctypes_bytes(data)
            bufsz = len(data)
        # Determine buffer mode
        if buffer_modes and i < len(buffer_modes):
            mode = buffer_modes[i]
        else:
            mode = cl.mem_flags.READ_ONLY
        if mode == cl.mem_flags.WRITE_ONLY:
            cl_buf = cl.Buffer(ctx, mode, size=bufsz)
        else:
            cl_buf = cl.Buffer(ctx, mode | cl.mem_flags.COPY_HOST_PTR, bufsz, hostbuf=ct_arr)
        cl_bufs.append(cl_buf)
    # Output buffer
    cl_output = cl.Buffer(ctx, output_mode, output_size)
    cl_bufs.append(cl_output)
    args = cl_bufs
    prg.__getattr__(function_name)(queue, (1,), None, *args)
    queue.finish()
    result_arr = c_uint64_4()
    cl.enqueue_copy(queue, result_arr, cl_output).wait()
    return ctypes_to_list(result_arr)

"""CPU reference: add two 256-bit values mod 2^256, limb by limb."""
def ref_add_256(a, b):
    result = [0, 0, 0, 0]
    carry = 0
    for i in range(4):
        s = int(a[i]) + int(b[i]) + carry
        result[i] = s & 0xFFFFFFFFFFFFFFFF
        carry = s >> 64
    return result

"""CPU reference: subtract two 256-bit values mod 2^256, limb by limb."""
def ref_sub_256(a, b):
    result = [0, 0, 0, 0]
    borrow = 0
    for i in range(4):
        d = int(a[i]) - int(b[i]) - borrow
        result[i] = d & 0xFFFFFFFFFFFFFFFF
        borrow = 1 if d < 0 else 0
    return result

"""CPU reference: reduce x mod p = 2^255 - 19 using Python pow."""
def ref_mod_p_reduce(x):
    """Reference: Python big-int modulo P. Ground truth."""
    full_value = sum(int(x[i]) << (64 * i) for i in range(4))
    result_val = full_value % P
    return to_256bit_array(result_val)

"""CPU reference: multiply two 256-bit values mod p using Python pow."""
def ref_mul_mod_p(a, b):
    a_val = sum(int(a[i]) << (64*i) for i in range(4))
    b_val = sum(int(b[i]) << (64*i) for i in range(4))
    result_val = (a_val * b_val) % P
    return to_256bit_array(result_val)


"""CPU reference: compute modular inverse a^(p-2) mod p using Python pow."""
def ref_mod_p_inverse(a):
    a_val = sum(int(a[i]) << (64*i) for i in range(4))
    result_val = pow(a_val, P-2, P)
    return to_256bit_array(result_val)


"""CPU reference: convert 32-byte LE seed to 4x64-bit scalar."""
def ref_seed_to_scalar(seed):
    result = [0, 0, 0, 0]
    for i in range(32):
        result[i // 8] |= (seed[i] << ((i % 8) * 8))
    return result

"""CPU reference: convert 4x64-bit scalar to 32-byte LE list."""
def ref_scalar_to_bytes(scalar):
    """CPU reference: scalar_to_bytes -- LITTLE-ENDIAN byte order.
    limb[0] (LSB) -> bytes[0..3], limb[3] (MSB) -> bytes[24..31]"""
    result = []
    for i in range(4):
        result.append((scalar[i] >>  0) & 0xFF)
        result.append((scalar[i] >>  8) & 0xFF)
        result.append((scalar[i] >> 16) & 0xFF)
        result.append((scalar[i] >> 24) & 0xFF)
    return result

"""Test GPU add_256 against CPU reference for random inputs."""
def test_add_256(kernel_code):
    print("Testing add_256...")
    test_cases = [
        ("0 + 0", [0, 0, 0, 0], [0, 0, 0, 0]),
        ("0 + 1", [0, 0, 0, 0], to_256bit_array(1)),
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
        expected = ref_add_256(a, b)
        result = run_opencl_test(kernel_code, "add_256", [(a, ctypes.c_uint64), (b, ctypes.c_uint64)], 32, [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
        if result == expected:
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {from_256bit_array(expected)}")
            print(f"    Got:      {from_256bit_array(result)}")
            all_passed = False
    return all_passed

"""Test GPU sub_256 against CPU reference for random inputs."""
def test_sub_256(kernel_code):
    print("Testing sub_256...")
    test_cases = [
        ("0 - 0", [0, 0, 0, 0], [0, 0, 0, 0]),
        ("1 - 0", to_256bit_array(1), [0, 0, 0, 0]),
        ("0 - 1", [0, 0, 0, 0], to_256bit_array(1)),
        ("1 - 1", to_256bit_array(1), to_256bit_array(1)),
        ("2 - 1", to_256bit_array(2), to_256bit_array(1)),
        ("2^255 - 2^255", to_256bit_array(2**255), to_256bit_array(2**255)),
        ("2^256-1 - 1", to_256bit_array(2**256 - 1), to_256bit_array(1)),
        ("2^256-1 - (2^256-1)", to_256bit_array(2**256 - 1), to_256bit_array(2**256 - 1)),
        ("123456789 - 987654321", to_256bit_array(123456789), to_256bit_array(987654321)),
        ("111111111 - 222222222", to_256bit_array(111111111), to_256bit_array(222222222)),
        ("999999999 - 111111111", to_256bit_array(999999999), to_256bit_array(111111111)),
        ("555555555 - 666666666", to_256bit_array(555555555), to_256bit_array(666666666)),
        ("777777777 - 888888888", to_256bit_array(777777777), to_256bit_array(888888888)),
    ]
    all_passed = True
    for name, a, b in test_cases:
        expected = ref_sub_256(a, b)
        result = run_opencl_test(kernel_code, "sub_256", [(a, ctypes.c_uint64), (b, ctypes.c_uint64)], 32, [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
        if result == expected:
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
"""Test GPU copy_256 copies all 4 limbs correctly."""
def test_copy_256(kernel_code):
    print("Testing copy_256...")
    test_cases = [
        ("copy zeros", [0, 0, 0, 0]),
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
        dst = [0, 0, 0, 0]
        result = run_opencl_test_two_out(kernel_code, "copy_256", (src, ctypes.c_uint64), (dst, ctypes.c_uint64), cl.mem_flags.READ_WRITE)
        if result == src:
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {src}")
            print(f"    Got:      {dst}")
            all_passed = False
    return all_passed
"""Test GPU one_256 sets value to [1, 0, 0, 0]."""
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
        result = [0, 0, 0, 0]
        result = run_opencl_test_single_inout(kernel_code, "one_256", (result, ctypes.c_uint64))
        if result == expected:
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
    expected = [0, 0, 0, 0]
    for name in test_cases:
        result = [0, 0, 0, 0]
        result = run_opencl_test_single_inout(kernel_code, "zero_256", (result, ctypes.c_uint64))
        if result == expected:
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
"""Test GPU mod_p_reduce against Python mod p reference."""
def test_mod_p_reduce(kernel_code):
    print("Testing mod_p_reduce...")
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
        expected = ref_mod_p_reduce(input_val)
        result = [0, 0, 0, 0]
        result = run_opencl_test_single_inout(kernel_code, "mod_p_reduce", (input_val, ctypes.c_uint64))
        if result == expected:
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
"""Test GPU mul_mod_p against Python (a*b) mod p reference."""
def test_mul_mod_p(kernel_code):
    print("Testing mul_mod_p...")
    test_cases = [
        ("0 * anything", [0, 0, 0, 0], to_256bit_array(12345)),
        ("anything * 0", to_256bit_array(12345), [0, 0, 0, 0]),
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
        expected = ref_mul_mod_p(a, b)
        result = run_opencl_test(kernel_code, "mul_mod_p", [(a, ctypes.c_uint64), (b, ctypes.c_uint64)], 32, [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
        if result == expected:
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    return all_passed
"""Test GPU mod_p_inverse against Python pow(a, p-2, p) reference."""
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
        expected = ref_mod_p_inverse(a)
        dst = ctypes.c_uint64 * 4
        result = run_opencl_test_two_out(kernel_code, "mod_p_inverse", (a, ctypes.c_uint64), (dst(0, 0, 0, 0), ctypes.c_uint64))
        if result == expected:
            print(f"  PASS: {name} {result}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            all_passed = False
    # Additional test: verify a * a^(-1) = 1
    if all_passed:
        for i, (name, a, _) in enumerate(test_cases[2:7], 2):
            dst = ctypes.c_uint64 * 4
            inv = run_opencl_test_two_out(kernel_code, "mod_p_inverse", (a, ctypes.c_uint64), (dst(0, 0, 0, 0), ctypes.c_uint64))
            product = run_opencl_test(kernel_code, "mul_mod_p", [(a, ctypes.c_uint64), (inv, ctypes.c_uint64)], 32)
            expected_one = to_256bit_array(1)
            if product == expected_one:
                print(f"  PASS: a * a^(-1) = 1 for test {i}")
            else:
                print(f"  FAIL: a * a^(-1) = 1 for test {i}")
                print(f"    a = {a}")
                print(f"    a^(-1) = {inv}")
                print(f"    a * a^(-1) = {product}")
                all_passed = False
    return all_passed
"""Test GPU seed_to_scalar against CPU byte-to-limb conversion."""
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
        scalar_ctypes = ctypes.c_uint64 * 4
        scalar = scalar_ctypes(0, 0, 0, 0)
        scalar = run_opencl_test_two_out(kernel_code, "seed_to_scalar", (seed_bytes, ctypes.c_uint8), (scalar, ctypes.c_uint64))
        expected = [0, 0, 0, 0]
        for i in range(32):
            expected[i // 8] |= (seed_bytes[i] << ((i % 8) * 8))
        result_list = ctypes_to_list(scalar)
        if result_list == expected:
            print(f"  PASS: {name} {result_list}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result_list}")
            all_passed = False
    return all_passed
"""Test GPU scalar_to_bytes against CPU limb-to-byte conversion."""
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
        bytes_out = bytes([0] * 32)
        result = run_opencl_test_two_out(kernel_code, "scalar_to_bytes", (scalar, ctypes.c_uint64), (bytes_out, ctypes.c_uint8))
        # Expected: LITTLE-ENDIAN byte order (consistent with seed_to_scalar)
        expected = [0] * 32
        for i in range(4):
            expected[i*8 + 0] = (scalar[i] >>  0) & 0xFF
            expected[i*8 + 1] = (scalar[i] >>  8) & 0xFF
            expected[i*8 + 2] = (scalar[i] >> 16) & 0xFF
            expected[i*8 + 3] = (scalar[i] >> 24) & 0xFF
        result_bytes = bytes(result)
        expected_bytes = bytes(expected)
        if result_bytes == expected_bytes:
            print(f"  PASS: {name} {result_bytes.hex()}")
        else:
            print(f"  FAIL: {name}")
            print(f"    Expected: {expected_bytes.hex()}")
            print(f"    Got:      {result_bytes.hex()}")
            all_passed = False
    return all_passed
"""Entry point: parse args, run selected tests, report results."""
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
            expected = ref_add_256(a_arr, b_arr)
            result = run_opencl_test(kernel_code, "add_256", [(a_arr, ctypes.c_uint64), (b_arr, ctypes.c_uint64)], 32,
                                    [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
            if result == expected:
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
            expected = ref_sub_256(a_arr, b_arr)
            result = run_opencl_test(kernel_code, "sub_256", [(a_arr, ctypes.c_uint64), (b_arr, ctypes.c_uint64)], 32,
                                    [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
            if result == expected:
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
            dst = ctypes.c_uint64 * 4
            expected = list(a_arr)
            result = run_opencl_test_two_out(kernel_code, "copy_256", (a_arr, ctypes.c_uint64), (dst(0, 0, 0, 0), ctypes.c_uint64), cl.mem_flags.READ_WRITE)
            if result == expected:
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
            result_ctypes = ctypes.c_uint64 * 4
            result = result_ctypes(0, 0, 0, 0)
            expected = to_256bit_array(1)
            result = run_opencl_test_single_inout(kernel_code, "one_256", (result, ctypes.c_uint64))
            if result == expected:
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
            result_ctypes = ctypes.c_uint64 * 4
            result = result_ctypes(0, 0, 0, 0)
            expected = [0, 0, 0, 0]
            result = run_opencl_test_single_inout(kernel_code, "zero_256", (result, ctypes.c_uint64))
            if result == expected:
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
            expected = ref_mod_p_reduce(a_arr)
            result_ctypes = ctypes.c_uint64 * 4
            result = result_ctypes(0, 0, 0, 0)
            result = run_opencl_test_single_inout(kernel_code, "mod_p_reduce", (a_arr, ctypes.c_uint64))
            if result == expected:
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
            expected = ref_mul_mod_p(a_arr, b_arr)
            result = run_opencl_test(kernel_code, "mul_mod_p", [(a_arr, ctypes.c_uint64), (b_arr, ctypes.c_uint64)], 32,
                                    [cl.mem_flags.READ_ONLY, cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY])
            if result == expected:
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
            expected = ref_mod_p_inverse(a_arr)
            result_ctypes = ctypes.c_uint64 * 4
            result = result_ctypes(0, 0, 0, 0)
            result = run_opencl_test_two_out(kernel_code, "mod_p_inverse", (a_arr, ctypes.c_uint64), (result, ctypes.c_uint64))
            if result == expected:
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
            scalar_ctypes = ctypes.c_uint64 * 4
            scalar = scalar_ctypes(0, 0, 0, 0)
            expected = ref_seed_to_scalar(seed_bytes)
            scalar = run_opencl_test_two_out(kernel_code, "seed_to_scalar", (seed_bytes, ctypes.c_uint8), (scalar, ctypes.c_uint64))
            if ctypes_to_list(scalar) == expected:
                print(f"  PASS")
                print(f"    Result: {from_256bit_array(ctypes_to_list(scalar))}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {from_256bit_array(expected)}")
                print(f"    Got:      {from_256bit_array(ctypes_to_list(scalar))}")
                sys.exit(1)
        elif args.func == "scalar":
            # Test scalar_to_bytes with custom a
            print(f"Testing scalar_to_bytes: {a_val}")
            a_arr = to_256bit_array(a_val)
            bytes_out = bytes([0] * 32)
            expected = ref_scalar_to_bytes(a_arr)
            result = run_opencl_test_two_out(kernel_code, "scalar_to_bytes", (a_arr, ctypes.c_uint64), (bytes_out, ctypes.c_uint8))
            result_bytes = bytes(result)
            expected_bytes = bytes(expected)
            if result_bytes == expected_bytes:
                print(f"  PASS")
                print(f"    Result: {result_bytes.hex()}")
            else:
                print(f"  FAIL")
                print(f"    Expected: {expected_bytes.hex()}")
                print(f"    Got:      {result_bytes.hex()}")
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
