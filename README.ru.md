# SSH-Ed25519-VanityGen
Генератор "красивых" (SSH-Vanity) ed25519-ключей.

<img src="https://img.icons8.com/emoji/24/000000/united-kingdom-emoji.png"/> [English readme](https://github.com/Aminuxer/SSH-Ed25519-VanityGen/blob/master/README.md)

Скрипты для генерации ключей Ed25519  с вашей строкой внутри тела ключа.
Ваши файлы authorized_keys станут чуть более читаемы; Коммент к ключу может коррелировать с телом ключа;
Дополнительная страховка от "фейковых ключей", лишь похожих на ваш.

## Версии
1). ssh_ed25519_vanity_multicpu.py - CPU-онли.
Многопроцессная версия на Python-3. Нагружает только процессорные ядра.

2). ssh_ed25519_vanity_gpu_opencl.py - GPU-[OpenCL]-версия.
Многопроцессная версия на Python-3. Нагружает только ядра OpenCL (видеокарты, вычислители FPGA ?).

## Установка и системные требования.

##### 0). Зависимости;

CPU-версия: Минимум **Python 3.6+**; **python3-cryptography 2.5+**;

GPU-версия: Минимум **Python 3.12+**; **python3-cryptography 3.1+**; **python3-pyopencl 2026.1+**;


Установите python3-пакет *python3-cryptography* с помощаью пакетного менеджера ОС или утилиты pip3.
```
dnf install python3-cryptography
or
apt install python3-cryptography
or 
pip3 install python3-cryptography
```

##### 1). Для версии под GPU-OpenCL (видеокарты, FPGA) также установите python3-pyopencl.
```
dnf install python3-opencl
or
apt install python3-opencl
or
pip3 install python3-opencl
```

Если версии пакетов в вашей системе слышком старые, попробуйте сперва скрипт **_pip-install-deps.sh**.

Хорошей практикой считается запуск ванити-генератора от имени отдельного пользователя в системе.

#### Проверка версий
```
# только CPU версия
python3 -c "import sys; print('Python:', sys.version)"
python3 -c "import cryptography; print('cryptography:', cryptography.__version__)"

# GPU-версия
python3 -c "import pyopencl; print('pyopencl:', pyopencl.__version__)"
```

#### PIP3-установка
pip3 install "cryptography>=2.5"

Для GPU также дополнительно:
```
pip3 install "pyopencl>=2026.1"
```

##### 1). Запустите команду для генерации ключа:
Пример с использованием только CPU:
```
python3 ssh_ed25519_vanity_multicpu.py User
```

Пример для видеокарты:
```
python3 ssh_ed25519_vanity_gpu_opencl.py User
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

_-w_ : Число потоков/вычислителей. По умолчанию = число ядер CPU или число видеокарт.

_-o_ : Выходной файл(ы) для найденных ключей. (префикс пути и имени)

_--debug_ : Печатать HEX-зерно (секретные байты) для отладки.

_--patterns-file_ : Файл со списком шаблонов, по одному на строку.


Специфичные для GPU-версии:

_--opencl-devices a,b,c_     Использовать только заданные OpenCL-устройства по их номеру (опция -w игнорируется)

_--load-percent 1-100_       Какой % микроядер каждого GPU будет использован (по умолчанию: 100)


Опции должны указываться после шаблона.

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


* Столкнулся со сложностями подготовки окружения.
  - Попробуйте установить pip-зависимости в профиль пользователя с помощью _pip-install-deps.sh
  - Попробуйте сделать отдельный venv профиль с помощью _pip-install-deps_manual_venv.sh
  - Попробуйте использовать podman/docker с помощью скриптов run-podman-gpu.sh / run-podman-cpu.sh

* Как запускать контейнеры ? Чем они отличаются ?
  - OpenCL-GPU/docker-env/run-podman-gpu.sh - GPU-версия, пробует детектить контейнерную среду (podman/docker), проивзодителя GPU и семейство ОС (разные glibc / opencl драйверы).
    Запускает вычисления на GPU в контейнере. Смотрите podman images -a / podman ps -a с именами вида 25519*gpu
  - docker-env/run-podman-cpu.sh CPU-версия, собирает контейнер из Dockerfile.cpu на основе python3.14-slim
    Запускает вычисления на CPU в контейнере. Смотрите podman images -a / podman ps -a с именами вида 25519*cpu
  - docker-env/run-podman-cpu-alpine.sh CPU-версия, собирает контейнер из Dockerfile.cpu-alpine на основе alpine-linux (библиотека musl)
    Запускает вычисления на CPU в контейнере с альпин-линуксом. Смотрите podman images -a / podman ps -a с именами вида 25519*cpu-alpine
  Эти контейнеры проще всего запускать через sh-скрипты для корректной работы OpenCL-библиотек и драйверов, маппинга путей и файловых параметров.

* Как быстро проверяются ключи ?

##### CPU:

| Процессор | Ядер использовано (-w) | Ядер всего | ~ ключей в секунду |
---|---|---|---|
| Celeron 633 [env-ed25519-translation] | 1 | 1 | 33 |
| Celeron 633 [native x86 python 3.8.10] | 1 | 1 | 261 |
| Celeron 433 [native x86 python 3.8.10] | 1 | 1 | 275 |
| AMD Turion II Neo N40 | 2 | 2 | 8 000 |
| Raspberry Pi 4 Model B Rev 1.5 [ARM]  | 4 | 4 | 10 000 |
| Core i5 CPU 650 3.20 GHz  | 4 | 4 | 22 000 |
| Celeron G9300 2.80 GHz | 2 | 2 | 30 000 |
| Xeon X3450 2.67 GHz | 8 | 8 | 37 000 |
| AMD Opteron(tm) Processor 2386 SE | 8 | 8 | 44 000 |
| Core-i7 2700K 3.40 GHz | 8 | 8 | 45 000 |
| Core i7-6700HQ 2.60 GHz  | 8 | 8 |  63 000  |
| Core i7-7700K 4.50 GHz  | 8 | 8 |  67 000  |
| Apple Silicon M4 CPU [ARM]  | 10 | 10 |  75 000  |
| Xeon CPU E5-2670 2.60 GHz | 20 | 32 | 140 000 |
| 2x Xeon CPU E5-2667 v4 3.60 GHz  | 32 | 32 | 232 000 |
| 2x AMD EPYC 7502 3.35 GHz | 128 | 128 | 315 000 |
| 2x AMD EPYC 9334 3.90 GHz | 128 | 128 | 640 000 |

##### GPU:

| Видеокарта | CU * WG | Ядрышек | ~ ключей в секунду |
---|---|---|---|
| GTX 1070 | 15 * 1024 | 15360 | 320 000 |
| GTX 1080 Ti | 28 * 1024 | 28672 | 590 000 |
| H100 NVL | 132 * 1024 | 135168 | 5 170 000 |

* Сколько ключей надо перебрать, чтобы найти желаемый ?
  - Можно посчитать мат-ожидание по формуле :
     C = ( ln(1/(1‑P)) * 64^N ) / ( 45 ‑ N )
    
     $$ C = \frac{64^{N} \ln \bigl(\frac{1}{1-P}\bigr)}{45 - N} $$
    
     Таблица оценки числа ключей для поиска с заданной вероятностью успеха:

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

* У меня очень старое железо (i686) и питон тоже очень древний (3.6).
  - Запустите CPU-верисю с опцией --debug для бенчмарка;
  - Используйте полученный seed (это секретные данные !!) для генерации ключа с помощью **seed-2-openssh-key.py** на более внятном оборудовании;


* Это реально 100%-ый вайб-кодинг ?
  - Да. QWEN-Coder-Next 80B/3B

* Что насчёт криптостойкости ?
  - Полагаемся на Ed25519 и функцию python3 os.urandom().
