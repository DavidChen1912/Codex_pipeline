## System Dependencies (Ubuntu 20.04 / 22.04)

### 1. System dependencies (Ubuntu)

Before creating the Python virtual environment, please install:

```bash
sudo apt update
sudo apt install -y \
  python3.10 \
  python3.10-venv \
  python3.10-dev \
  build-essential \
  libopenslide0 \
  libopenslide-dev \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1
```

### 2. Create virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Upgrade Python build tools (recommended)

```bash
pip install -U pip setuptools wheel
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```
