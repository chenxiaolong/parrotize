Parrotize
---------

A Python script to party-parrotize any image.

Dependencies
------------

Parrotize depends on the ImageMagick native library to run. It can be installed via the following commands.

### macOS

```sh
brew install imagemagick
```

### Fedora

```sh
sudo dnf install ImageMagick
```

### RHEL-based distros

```sh
sudo yum install ImageMagick
```

### Debian-based distros

```sh
sudo apt install imagemagick
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