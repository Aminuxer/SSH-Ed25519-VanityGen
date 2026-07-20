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
Установите python3-пакет *python3-cryptography* с помощаью пакетного менеджера ОС или утилиты pip3.
```
dnf install python3-cryptography
or
apt install python3-cryptography
or 
pip3 install python3-cryptography
```


1). Запустите команду для генерации ключа:
Пример с использование только CPU:
```
python3 ssh_ed25519_vanity_multicpu.py User
```

Пример вывода:


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

## Опции
_-i_ : Нечувствительность к регистру. Сильно ускоряет поиск ключа, но регистр символов вряд ли совпадёт.


_-w_ : Число потоков. По умолчанию = число ядер CPU.


_-o_ : Выходноый файл(ы) для найденных ключей.


Опции должны указывать после шаблона.

Пример команды:
```
python3 ssh_ed25519_vanity_multicpu.py User -w 6 -i -o user_key
```
Пример вывода:
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
* Можно ли указать больше потоков, чем ядер ?
  - Да. Можно попробовать разные варианты.

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
