Parrotize
---------

A Python script to party-parrotize any image.

Dependencies
------------

Parrotize depends on the ImageMagick native library to run. It can be installed via the OS's package manager.

Note that version 7 is required. Version 6 will produce output files with some single-color blocky artifacts. If you're on Linux and your distro does not package version 7, the official AppImage can be used.

First, download the AppImage from: https://imagemagick.org/script/download.php

Then, extract it and set the `MAGICK_HOME` environment variable to point to the extracted files.

```bash
./magick --appimage-extract
export MAGICK_HOME=$(pwd)/squashfs-root/usr
```

Usage
-----

Set up a virtualenv and install the Python dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To parrotize an image (eg. `foobar.png`), simply run:

```sh
python parrotize.py foobar.png foobar.gif
```

Additional options (speed control, etc.) can be found in:

```sh
python parrotize.py --help
```
