# SSH-Ed25519-VanityGen
Генератор "красивых" (SSH-Vanity) ed25519-ключей.

<img src="https://img.icons8.com/emoji/24/000000/united-kingdom-emoji.png"/> [English readme](https://github.com/Aminuxer/SSH-Ed25519-VanityGen/blob/master/README.md)

Скрипты для генерации ключей Ed25519  с вашей строкой внутри тела ключа.
Ваши файлы authorized_keys станут чуть более читаемы; Коммент к ключу может коррелировать с телом ключа;
Дополнительная страховка от "фейковых ключей", лишь похожих на ваш.

## Версии
1). ssh_ed25519_vanity_multicpu.py - CPU-онли.
Многопоточная версия на Python-3. Нагружает только процессорные ядра.

## Установка и системные требования.

0). Зависимости;
Минимальные требуемые версии: Python 3.6; python3-cryptography 2.5;

Установите python3-пакет *python3-cryptography* с помощаью пакетного менеджера ОС или утилиты pip3.
```
dnf install python3-cryptography
or
apt install python3-cryptography
or 
pip3 install python3-cryptography
```

#### Проверка версий
```
python3 -c "import sys; print('Python:', sys.version)"
python3 -c "import cryptography; print('cryptography:', cryptography.__version__)"
```

#### PIP3-установка
pip3 install "cryptography>=2.5"


1). Запустите команду для генерации ключа:
Пример с использование только CPU:
```
python3 ssh_ed25519_vanity_multicpu.py User
```

Пример вывода:


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

## Опции
_-i_ : Нечувствительность к регистру. Сильно ускоряет поиск ключа, но регистр символов вряд ли совпадёт.
_-w_ : Число потоков. По умолчанию = число ядер CPU.
_-o_ : Выходной файл(ы) для найденных ключей.
_--debug_ : Печатать HEX-зерно (секретные байты) для отладки.
_--patterns-file_ : Файл со списком шаблонов, по одному на строку.


Опции должны указывать после шаблона.

Пример команды:
```
python3 ssh_ed25519_vanity_multicpu.py User -w 6 -i -o user_key
```
Пример вывода:
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


Поиск по нескольким шаблонам сразу:
```
cat patterns-list.txt
Use+
USE+
Use-
-US-
/US+
```

Пример вывода:
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
[+] Written: KEYS-Use_-20260724-223234.pub and KEYS-Use_-2026... (mode 600)
[*] Continuing search for remaining patterns...

[+] Found match for '/US+'!                                 \   /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFrJnfh1/US+XNiYaNJXMFv9ytKxUuJmSeYx7nv1yuMx /US+
[+] Written: KEYS-_US_-20260724-223237.pub and KEYS-_US_-2026... (mode 600)
[*] Continuing search for remaining patterns...

[+] Found match for 'USE+'!                                                              \    /
[+] Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBkZmpR+3nmBzrATC1lYcPSJUnb/OZBfOkNIWCUSE+nz USE+
[+] Written: KEYS-USE_-20260724-223238.pub and KEYS-USE_-2026... (mode 600)
[*] Continuing search for remaining patterns...

[+] Checked keys: 35,004 (avg ~6,903 keys/sec)
[+] Total time: 5s
[+] Checked keys: 35,004
```

##  FAQ
* Можно ли указать больше потоков, чем ядер ?
  - Да. Можно попробовать разные варианты.

* Будет ли добавлена поддержка регулярных выражений для шаблонов ?
  - Точно нет. Они слишком медленные.

* Как быстро проверяются ключи ?
    ```
    ~ 30 тысяч ключей в секунду на двухядерном Celeron G9300 2.80 GHz (on 2-x cores).
    ~ 45 тысяч ключей в секунду на Core-i7 2700K (on 8-x cores).
    ~ 130 тысяч ключей в секунду на Xeon(R) CPU E5-2670 0 @ 2.60GHz (on 20-х cores)
    ~ 550 тысяч ключей в секунду на  AMD EPYC 9334 (on 128-х cores)
    ```
* Это реально 100%-ый вайб-кодинг ?
  - Да. QWEN-Coder-Next 80B/3B

* Что насчёт криптостойкости ?
  - Полагаемся на Ed25519 и функцию python3 os.urandom().
