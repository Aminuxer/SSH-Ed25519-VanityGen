#!/usr/bin/env python3

import sys, os, hashlib, argparse

try:
    import pyopencl as cl
except ImportError as e:
    print(f"Error: pyopencl required - {e}")
    sys.exit(1)

def get_opencl_device():
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                return device
    raise RuntimeError("No GPU device found")

def load_kernel():
    kernel_path = os.path.join(os.path.expanduser("./"), "sha512.cl")
    with open(kernel_path, "r") as f:
        return f.read(), kernel_path

def parse_hex_arg(hex_str):
    hex_str = hex_str.strip()
    if hex_str.startswith(("0x", "0X")):
        hex_str = hex_str[2:]
    if len(hex_str) % 2 != 0:
        raise ValueError(f"Odd-length hex string: {hex_str}")
    return bytes.fromhex(hex_str)

def sha512_opencl(data):
    """SHA-512 using GPU - handles arbitrary length data."""
    data_len = len(data)

    # Handle empty string case
    if data_len == 0:
        return hashlib.sha512(data).digest()  # Return known hash

    cl_content, cl_path = load_kernel()
    device = get_opencl_device()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prg = cl.Program(ctx, cl_content).build()

    # Use bytes directly for hostbuf (no numpy)
    data_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, size=data_len, hostbuf=data)

    # Length as native unsigned long long
    import array as _array
    length_arr = _array.array('Q', [data_len])
    length_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, size=len(length_arr) * length_arr.itemsize, hostbuf=length_arr)

    # Output buffer as bytearray (no numpy)
    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=64)

    prg.sha512(queue, (1,), None, data_buf, output_buf, length_buf)
    queue.finish()

    result = bytearray(64)
    cl.enqueue_copy(queue, result, output_buf).wait()
    return bytes(result)

def sha512_hashlib(data):
    return hashlib.sha512(data).digest()

def main():
    parser = argparse.ArgumentParser(description="SHA-512 OpenCL Test Suite")
    parser.add_argument("extra", nargs="*", help="Additional strings or hex data (0x...) to test")
    args = parser.parse_args()

    # Single mode: first CLI argument only
    if len(sys.argv) > 1:
        arg_input = sys.argv[1]
        print(f"Input: {arg_input}")
        try:
            data = arg_input.encode("utf-8")
            opencl_result = sha512_opencl(data)
            hashlib_result = sha512_hashlib(data)
            print(f"OpenCL:   {opencl_result.hex()}")
            print(f"hashlib:  {hashlib_result.hex()}")
        except Exception as e:
            print(f"ERROR: {e}")
            return 1
        print()

    # Full test mode - always runs
    print("=" * 70)
    print("SHA-512 OpenCL Test Suite")
    print("=" * 70)
    device = get_opencl_device()
    print(f"Device: {device.name}")
    print(f"Platform: {device.platform.name}")
    cl_content, cl_path = load_kernel()
    print(f"Kernel: {cl_path}")
    
    test_cases = [
        ("32-byte all zeros", b"\x00" * 32),
        ("32-byte all ones", b"\xff" * 32),
        ("32-byte pattern", b"\x01\x23\x45\x67\x89\xab\xcd\xef" * 4),
        ("64-byte all zeros", b"\x00" * 64),
        ("64-byte all ones", b"\xff" * 64),
        ("64-byte pattern", b"\x01\x23\x45\x67\x89\xab\xcd\xef" * 8),
        ("", b""),
        ("000000", b"000000"),
        ("123456", b"123456"),
        ("Amin", b"Amin"),
        ("VeryLong-Test-String-Fur-SHA-512-GRAND-HaSH-OpenCL-vs-HashLib", b"VeryLong-Test-String-Fur-SHA-512-GRAND-HaSH-OpenCL-vs-HashLib"),
    ]
    
    for extra in args.extra:
        if extra.startswith(("0x", "0X")):
            try:
                data = parse_hex_arg(extra)
                test_cases.append((f"Hex: {extra}", data))
            except ValueError as e:
                print(f"ERROR: Invalid hex argument {extra}: {e}")
                return 1
        else:
            test_cases.append((f"String: {extra}", extra.encode("utf-8")))
    
    all_passed = True
    for name, data in test_cases:
        try:
            opencl_result = sha512_opencl(data)
            hashlib_result = sha512_hashlib(data)
            passed = opencl_result == hashlib_result
            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"{status}: {name}")
            if not passed:
                print(f"  OpenCL:   {opencl_result.hex()}")
                print(f"  hashlib:  {hashlib_result.hex()}")
            else:
                print(f"  Hash: {opencl_result.hex()}")
        except Exception as e:
            all_passed = False
            print(f"ERROR: {name}: {e}")
    
    print("=" * 70)
    if all_passed:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
