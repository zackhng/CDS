# CDS
CDS 50.038


## Environment Setup

This project uses a **Python virtual environment (`venv`)** to manage dependencies.  
The `venv/` folder is included in `.gitignore`, so it is **not stored in the repository**. Each user must recreate the environment locally using the `requirements.txt` file.

### Prerequisites

Ensure the following are installed:

- Python 3.8 or higher
- `pip` (Python package manager)

Check your Python version:

```bash
python3 --version
````

---

### 1. Create the virtual environment

From the root directory of the project, run:

```bash
python3 -m venv venv
```

This creates a folder named `venv/` in the project directory.

Project structure example:

```text
project/
├── venv/
├── requirements.txt
├── src/
└── README.md
```

---

### 2. Activate the virtual environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

After activation, your terminal should show:

```text
(venv)
```

---

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

> Note: Additional packages may be installed automatically as **dependencies** of the listed packages.

---

### 4. Verify installation (optional)

```bash
pip list
```

---

### 5. Using the environment in VS Code

1. Open the Command Palette

   * macOS: `Cmd + Shift + P`
   * Windows/Linux: `Ctrl + Shift + P`

2. Search for:

```text
Python: Select Interpreter
```

3. Select the interpreter located at:

```text
./venv/bin/python
```

VS Code will now use this environment for running and debugging the project.

---

### 6. Deactivating the environment

When you are done working, deactivate the virtual environment with:

```bash
deactivate
```

