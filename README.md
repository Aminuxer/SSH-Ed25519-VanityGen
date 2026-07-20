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
Install python3 package *python3-cryptography* with OS package manager or over pip3 tool.
```
dnf install python3-cryptography
or
apt install python3-cryptography
or 
pip3 install python3-cryptography
```

1). Run command for generate key:
On CPU only:
```
python3 ssh_ed25519_vanity_multicpu.py User
```

Example output

```
[*] Searching for pattern: User
[*] Case insensitive: False
[*] Using 8 worker processes

[+] Found match!
[+] Pattern: User                                                                    \    /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOgoT3M3vgzd9RuPqE4cS5v8xdjtHbY8CKUserCvVGxc
[+] Seed (hex): 3c0ddad9f6b80269a290818270fc5480952deada4f067f8468425e021aef25e4     /    \

-----BEGIN OPENSSH PRIVATE KEY-----
...
-----END OPENSSH PRIVATE KEY-----

[+] Total time: 5s
```

## Options
*-i* : Case insensitivity.
*-w* : Workers (thread) count. By default = count of CPU cores.
*-o* : Output file(s) for founded key.
Options must be specified after pattern.
Example command:
```
python3 ssh_ed25519_vanity_multicpu.py User -w 6 -i -o user_key
```
Example Output:
```
[*] Searching for pattern: User
[*] Case insensitive: True
[*] Using 6 worker processes

[+] Found match!
[+] Pattern: User                                                                     \    /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINjlM36Apehfaws+7SePQQXLha1142WsZlsuSerI7+r+
[+] Seed (hex): 4e068d11efbd54912780f29426008c5c0f6d6aed3e2852e8266dd67512d89e0d      /    \

-----BEGIN OPENSSH PRIVATE KEY-----
...
-----END OPENSSH PRIVATE KEY-----

[+] Written: user_key.pub
[+] Written: user_key (mode 600)
[+] Total time: 0s
```

##  FAQ
* Can i specify more workers then vCPU ?
  - Yes. You can try different variants.

* How fast keys checked ?
    ```
    ~ 30 000 keys/sec Celeron G9300 2.80 GHz (on 2-x cores).
    ~ 45 000 keys/sec Core-i7 2700K (on 8-x cores).
    ~ 130 000 keys/sec Xeon(R) CPU E5-2670 0 @ 2.60GHz (on 20-х cores)
    ~ 550 000 keys/sec  AMD EPYC 9334 (on 128-х cores)
    ```

* It's really 100% vibe coding ?
  - Yes. QWEN-Coder-Next 80B/3B

* What's about cpyprographic reliability ?
  - Rely on Ed25519 and puython3 os.urandom() quality.

