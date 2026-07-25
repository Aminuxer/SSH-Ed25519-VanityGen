# SSH-Ed25519-VanityGen
SSH Vanity ed25519 key generator.

<img src="https://img.icons8.com/emoji/24/000000/russia-emoji.png"/> [описание на русском](https://github.com/Aminuxer/SSH-Ed25519-VanityGen/blob/master/README.ru.md)

Scripts for generate vanity OpenSSH Ed-25519 keys with your desired string *INSIDE* public key body.
Your authorized_keys files will more readable; Comments can correlate with strings INSIDE key-body;
Additional defend against "fake-keys".

## Versions
1). ssh_ed25519_vanity_multicpu.py - CPU-version.
Python-3 multithreaded version. Load only CPU cores.

## Installation and requirements

0). Dependencies;
Minimal versions: Python 3.6; python3-cryptography 2.5


Install python3 package *python3-cryptography* with OS package manager or over pip3 tool.
```
dnf install python3-cryptography
or
apt install python3-cryptography
or 
pip3 install python3-cryptography
```
#### Check versions
```
python3 -c "import sys; print('Python:', sys.version)"
python3 -c "import cryptography; print('cryptography:', cryptography.__version__)"
```

#### PIP-install
pip3 install "cryptography>=2.5"


1). Run command for generate key:
On CPU only:
```
python3 ssh_ed25519_vanity_multicpu.py User
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

Options must be specified after pattern of pattern-file.
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

* How fast keys checked ?
    ```
    ~ 30 000 keys/sec Celeron G9300 2.80 GHz (on 2-x cores).
    ~ 45 000 keys/sec Core-i7 2700K (on 8-x cores).
    ~ 130 000 keys/sec Xeon(R) CPU E5-2670 0 @ 2.60GHz (on 20-х cores)
    ~ 550 000 keys/sec  AMD EPYC 9334 (on 128-х cores)
    ```
* How many keys need brute until found key ?
  - You can calucate math-estimate by formula :
     C = ( ln(1/(1‑P)) * 64^N ) / ( 44‑N+1 )
    
     $ \( C = \frac{64^{N}\,\ln\!\bigl(\frac{1}{1-P}\bigr)}{44 - N + 1} \) $
    
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


* It's really 100% vibe coding ?
  - Yes. QWEN-Coder-Next 80B/3B

* What's about cryprographic reliability ?
  - Rely on Ed25519 and puython3 os.urandom() quality.

