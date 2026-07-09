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
  - ~ 200K keys per second on Dual Core Celeron G9300. Up to 7-8M keys per second on Core-i7 2700K / Xeon.

* It's really 100% vibe coding ?
  - Yes. QWEN-Coder-Next 80B/3B

* What's about cpyprographic reliability ?
  - Rely on Ed25519 and puython3 os.urandom() quality.

