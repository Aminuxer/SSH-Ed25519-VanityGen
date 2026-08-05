# Ed-25519 SSH Vanity Key Generator [OpenCL GPU]

## Overview

This project implements an SSH ED25519 vanity key generator using **OpenCL GPU acceleration** for full key generation and pattern checking.

- **100% AI-Generated**: QWEN-3.6-27B NVFP4
- **Inspired by Aminuxer**
- https://github.com/Aminuxer/SSH-Ed25519-VanityGen/tree/master/OpenCL-GPU

---

## Dependencies

#### Required:
* python3-pyopencl >= 2026.1
* python3-numpy
* python3-cryptography >= 3.1


* OpenCL device (GPU) with drivers for OpenCL support


## Usage

```
python3 ssh_ed25519_vanity_gpu_opencl.py <pattern> [-i] [-w <workers>] [-o output] [--debug]
python3 ssh_ed25519_vanity_gpu_opencl.py --patterns-file <file> [-i] [-w <workers>] [-o output] [--debug]

python3 ssh_ed25519_vanity_gpu_opencl.py --help
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<pattern>` | Base64 pattern to search for in the public key (e.g., `User`, `Ami`) |
| `-i` | Case-insensitive pattern matching |
| `-w <workers>` | Number of GPU workers to use (default: number of detected GPUs) |
| `-o <output_file>` | Save results to file (creates `.pub` for public key, main file for private key) |
| `--debug` | Display source seed |

GPU-specific options:
```
  --opencl-devices a,b,c     Use specific device IDs (ignores -w)
  --load-percent 1-100       % of GPU cores to use (default: 100)
```

---

## Key Format

SSH ED25519 public key format:
```
ssh-ed25519 <base64_public_key> <comment>
```
