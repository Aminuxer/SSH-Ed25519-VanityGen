# SSH-Ed25519-VanityGen
SSH Vanity ed25519 key generator.

<img src="https://img.icons8.com/emoji/24/000000/russia-emoji.png"/> [описание на русском](https://github.com/Aminuxer/SSH-Ed25519-VanityGen/blob/master/README.ru.md)

Scripts for generate vanity OpenSSH Ed-25519 keys with your desired string *INSIDE* public key body.
Your authorized_keys files will more readable; Comments can correlate with strings INSIDE key-body;
Additional defend against "fake-keys".

## Versions
1). ssh_ed25519_vanity_multicpu.py - CPU-version.
Python-3 multiprocessing version. Load only CPU cores.

2). ssh_ed25519_vanity_gpu_opencl.py - GPU-[OpenCL]-version.
Python-3 multiprocessing + OpenCL (pyopencl) version. Load only OpenCL cores (GPU, FPGA ?).

## Installation and requirements

##### 0). Dependencies;

CPU-version:  **Python 3.6+**; **python3-cryptography 2.5+**

GPU-version: **Python 3.12+**; **python3-cryptography 3.1+**, **python3-pyopencl-2026.1+**;


Install python3 package *python3-cryptography* with OS package manager or over pip3 tool.
```
dnf install python3-cryptography
or
apt install python3-cryptography
or 
pip3 install python3-cryptography
```

##### 1). For GPU-OpenCL version install python3-pyopencl.
```
dnf install python3-opencl
or
apt install python3-opencl
or 
pip3 install python3-opencl
```
If your environment contain too old version, try use script **_pip-install-deps.sh**.

Running vanity-gen under separated user profile is good practice.

#### Check versions
```
# CPU-only version
python3 -c "import sys; print('Python:', sys.version)"
python3 -c "import cryptography; print('cryptography:', cryptography.__version__)"

# Also check for GPU version
python3 -c "import pyopencl; print('pyopencl:', pyopencl.__version__)"
```

#### PIP-install

```
pip3 install "cryptography>=2.5"
```


For GPU:
```
pip3 install "pyopencl>=2026.1"
```

##### 1). Run command for generate key:
On CPU only:
```
python3 ssh_ed25519_vanity_multicpu.py User
```

On GPU:
```
python3 ssh_ed25519_vanity_gpu_opencl.py User
```

Example output

```
[*] Accepted patterns: User
[*] Case insensitive: False
[*] Debug mode: False
[*] Using 8 worker processes

[+] Found match for 'User'!                                                          \    /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOgoT3M3vgzd9RuPqE4cS5v8xdjtHbY8CKUserCvVGxc User
[!] Output to console (file save skipped or failed):                                 /    \
-----BEGIN OPENSSH PRIVATE KEY-----
...
-----END OPENSSH PRIVATE KEY-----

[*] Continuing search for remaining patterns...

[+] Checked keys: 3208 (avg ~1152 keys/sec)
[+] Total time: 15s
[+] Checked keys: 3208
```

## Options

_-i_ : Case insensitivity.

_-w_ : Workers (thread) count. By default = count of CPU cores.

_-o_ : Output filename prefix for founded key(s).

_--debug_ : Print HEX-seed private bytes for debug.

_--patterns-file_ : Text file with patterns, one per string

GPU-specific options:

_--opencl-devices a,b,c_     Use specific device IDs (ignores -w)

_--load-percent 1-100_       % of GPU cores to use (default: 100)


Options must be specified after pattern or pattern-file.
Example command:

```
python3 ssh_ed25519_vanity_multicpu.py User -w 6 -i -o user_key
```
Example Output:
```
[*] Accepted patterns: User
[*] Case insensitive: True
[*] Debug mode: False
[*] Using 8 worker processes

[+] Found match for 'User'!                                                           \    /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINjlM36Apehfaws+7SePQQXLha1142WsZlsuSerI7+r+
[+] Written: user_key-User-202607....pub and user-key-User-202607...                  /    \
[*] Continuing search for remaining patterns...

[+] Checked keys: 1254 (avg ~1589 keys/sec)
[+] Total time: 20s
[+] Checked keys: 1254
```

Multi-patterns example:
```
cat patterns-list.txt
Use+
USE+
Use-
-US-
/US+
```

Example output:
```
python3 ssh_ed25519_vanity_multicpu.py --patterns-file patterns-list.txt -o KEYS
[-] Warning: Skipping invalid pattern: 'Use-'
[-] Warning: Skipping invalid pattern: '-US-'
[*] Accepted patterns: Use+, USE+, /US+
[*] Case insensitive: False
[*] Debug mode: False
[*] Using 8 worker processes

[+] Found match for 'Use+'!                                                  \    /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDy5zVZ/dic/CYpAA+JRFJvC+NUse+Z/Z3yXJdwMkGiS Use+
[+] Written: KEYS-Use_-2026....pub and KEYS-Use_-2026... (mode 600)
[*] Continuing search for remaining patterns...

[+] Found match for '/US+'!                                 \   /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFrJnfh1/US+XNiYaNJXMFv9ytKxUuJmSeYx7nv1yuMx /US+
[+] Written: KEYS-_US_-2026....pub and KEYS-_US_-2026... (mode 600)
[*] Continuing search for remaining patterns...

[+] Found match for 'USE+'!                                                              \    /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBkZmpR+3nmBzrATC1lYcPSJUnb/OZBfOkNIWCUSE+nz USE+
[+] Written: KEYS-USE_-2026....pub and KEYS-USE_-2026... (mode 600)
[*] Continuing search for remaining patterns...

[+] Checked keys: 35,004 (avg ~6,903 keys/sec)
[+] Total time: 5s
[+] Checked keys: 35,004
```


##  FAQ
* Can i specify more workers then vCPU ?
  - Yes. You can try different variants.

* Will be RegExp support added for patterns ?
  - No. Too slow function.

* I catch issues with environment prepare.
  - try to install pip-deps in user-profile with _pip-install-deps.sh
  - try to make venv profile with _pip-install-deps_manual_venv.sh
  - try use podman/docker with run-podman-gpu.sh / run-podman-cpu.sh

* How to run containers ? What difference ?
  - OpenCL-GPU/docker-env/run-podman-gpu.sh - GPU-version. try detect container toolkit (podman/docker), GPU vendor and OS family (different glibc / opencl driver).
    Run calculation on GPU in container. See podman images -a / podman ps -a with names 25519*gpu
  - docker-env/run-podman-cpu.sh CPU-version, build container from Dockerfile.cpu based on python3.14-slim
    Run calculation on CPU in container. See podman images -a / podman ps -a with names 25519*cpu
  - docker-env/run-podman-cpu-alpine.sh CPU-version, build container from Dockerfile.cpu-alpine based on alpine-linux (musl library)
    Run calculation on CPU in container. See podman images -a / podman ps -a with names 25519*cpu-alpine
  This containers easy to run over sh-scripts for correct mount OpenCL-files and drivers, path mapping and file-related options.

* How fast keys checked ?

##### CPU:

| Hardware | Cores used (-w) | Cores ALL | ~ keys per second |
---|---|---|---|
| Celeron 633 [env-ed25519-translation] | 1 | 1 | 33 |
| Celeron 633 [native x86 python 3.8.10] | 1 | 1 | 261 |
| Celeron 433 [native x86 python 3.8.10] | 1 | 1 | 275 |
| AMD Turion II Neo N40 | 2 | 2 | 8 000 |
| Raspberry Pi 4 Model B Rev 1.5 [ARM]  | 4 | 4 | 10 000 |
| Core i5 650 3.20 GHz  | 4 | 4 | 22 000 |
| Celeron G9300 2.80 GHz | 2 | 2 | 30 000 |
| Xeon X3450 2.67GHz | 8 | 8 | 37 000 |
| AMD Opteron(tm) Processor 2386 SE | 8 | 8 | 44 000 |
| Core-i7 2700K 3.40 GHz | 8 | 8 | 45 000 |
| Core i7-6700HQ 2.60 GHz  | 8 | 8 |  63 000  |
| Core i7-7700K 4.50 GHz  | 8 | 8 |  67 000  |
| Apple Silicon M4 CPU [ARM]  | 10 | 10 |  75 000  |
| Xeon CPU E5-2670 2.60 GHz  | 20 | 32 | 140 000 |
| 2x Xeon CPU E5-2667 v4 3.60 GHz  | 32 | 32 | 232 000 |
| 2x AMD EPYC 7502 3.35 GHz | 128 | 128 | 315 000 |
| 2x AMD EPYC 9334 3.90 GHz | 128 | 128 | 640 000 |

##### GPU:

| Hardware | CU * WG | Cores ALL | ~ keys per second |
---|---|---|---|
| GTX 1070 | 15 * 1024 | 15360 | 320 000 |
| GTX 1080 Ti | 28 * 1024 | 28672 | 590 000 |
| GTX 3060 | 28 * 1024 | 28672 | 800 000 |
| H100 NVL | 132 * 1024 | 135168 | 5 170 000 |

* How many keys need brute until found key ?
  - You can calucate math-estimate by formula :
     C = ( ln(1/(1‑P)) * 64^N ) / ( 45‑N )
    
     $$ C = \frac{64^{N} \ln \bigl(\frac{1}{1-P}\bigr)}{45 - N} $$
     or with normalization:
    $$ C(N, p) = 2^{256} \left[ 1 - \exp\left( -\frac{64^N \ln\!\left(\frac{1}{1-p}\right)}{(45 - N) \cdot 2^{256}} \right) \right] $$
    
     Table with estimated keys count for target probability:

| N \ P | 50 | 90 | 99 | 99,9 |
---|---|---|---|---|
| 2 | 66 | 219 | 439 | 658 |
| 3 | 4326 | 14372 | 28743 | 43115 |
| 4 | 283636 | 942219 | 1884437 | 2826656 |
| 5 | 18606528 | 61809548 | 123619096 | 185428644 |
| 6 | 1221351578 | 4057242121 | 8114484243 | 12171726364 |
| 7 | 80223514188 | 266496745652 | 532993491303 | 799490236955 |
| 8 | 5273069905545 | 17516759065535 | 35033518131070 | 52550277196607 |
| 9 | 346850820453631 | 1152213485199650 | 2304426970399298 | 3456640455599000 |
| 10 | 22832694009290500 | 75848567711428400 | 151697135422857000 | 227545703134289000 |

* My hardware too old (i686) and python too old also (3.6).
  - Run CPU-version with --debug option for benchmark;
  - Use seed (this data is secret !!) for generate keypair wuth **seed-2-openssh-key.py** on more fresh equipment;

* It's really 100% vibe coding ?
  - Yes. QWEN-Coder-Next 80B/3B

* What's about cryprographic reliability ?
  - Rely on Ed25519 and python3 os.urandom() quality.

