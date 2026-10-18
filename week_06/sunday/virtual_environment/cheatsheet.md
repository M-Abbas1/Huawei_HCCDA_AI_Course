# 🐍 Python Virtual Environment — Command Cheat Sheet

## 1. Check Python

```bash
python --version
```

Check whether Python is installed.

```bash
python -m pip --version
```

Check whether `pip` is available.

---

# 2. Create a Virtual Environment

### Windows / Linux / macOS

```bash
python -m venv .venv
```

Creates a virtual environment named `.venv`.

You can also choose another name:

```bash
python -m venv myenv
```

### Common naming convention

```text
.venv
venv
env
```

`.venv` is commonly used for project-specific environments.

---

# 3. Activate the Environment

## 🪟 Windows — Command Prompt (CMD)

```cmd
.venv\Scripts\activate
```

## 🪟 Windows — PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 🐧 Linux / macOS

```bash
source .venv/bin/activate
```

### After activation

Your terminal will usually show something like:

```text
(.venv) C:\my_project>
```

The `(.venv)` tells you that the virtual environment is active.

---

# 4. Deactivate the Environment

Works on Windows, Linux, and macOS:

```bash
deactivate
```

This returns you to the previous Python environment.

---

# 5. Check Which Python Is Being Used

### Windows

```cmd
where python
```

### Linux / macOS

```bash
which python
```

Or use Python itself:

```bash
python -c "import sys; print(sys.executable)"
```

If the environment is active, the path should point inside `.venv`.

Example:

```text
C:\my_project\.venv\Scripts\python.exe
```

---

# 6. Install Packages

After activating the environment:

```bash
python -m pip install numpy
```

Install multiple packages:

```bash
python -m pip install numpy pandas matplotlib
```

Install an AI library:

```bash
python -m pip install torch
```

```bash
python -m pip install tensorflow
```

---

# 7. Check Installed Packages

```bash
python -m pip list
```

Shows all packages installed in the active environment.

To see information about one package:

```bash
python -m pip show numpy
```

---

# 8. Check Package Version

```bash
python -c "import numpy; print(numpy.__version__)"
```

Or:

```bash
python -m pip show numpy
```

---

# 9. Upgrade a Package

```bash
python -m pip install --upgrade numpy
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

---

# 10. Uninstall a Package

```bash
python -m pip uninstall numpy
```

Python will ask for confirmation.

---

# 11. Install a Specific Version

```bash
python -m pip install numpy==1.26.4
```

General format:

```bash
python -m pip install package_name==version
```

Example:

```bash
python -m pip install tensorflow==2.18.0
```

---

# 12. Install a Minimum Version

```bash
python -m pip install "numpy>=1.26"
```

This means:

```text
NumPy version 1.26 or newer
```

---

# 13. Save Project Dependencies

After installing everything your project needs:

```bash
python -m pip freeze > requirements.txt
```

This creates:

```text
requirements.txt
```

Example:

```text
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
```

---

# 14. Install Dependencies from requirements.txt

First create and activate your environment:

```bash
python -m venv .venv
```

Activate it, then:

```bash
python -m pip install -r requirements.txt
```

This installs the packages listed in the file.

---

# 15. Create a New Environment from an Existing Project

Typical workflow:

```bash
git clone <project>
cd project
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Now the project environment is ready.

---

# 16. Delete a Virtual Environment

There is normally no special `delete` command.

First deactivate it:

```bash
deactivate
```

Then delete the `.venv` folder.

### Windows CMD

```cmd
rmdir /s /q .venv
```

### PowerShell

```powershell
Remove-Item -Recurse -Force .venv
```

### Linux / macOS

```bash
rm -rf .venv
```

⚠️ Be careful with `rm -rf` and `Remove-Item -Recurse -Force`. Make sure `.venv` is the folder you intend to delete.

---

# 17. Create Environment with a Specific Python Version

If multiple Python versions are installed, you can specify the interpreter.

### Windows example

```cmd
py -3.11 -m venv .venv
```

### Linux / macOS example

```bash
python3.11 -m venv .venv
```

This creates the environment using Python 3.11.

---

# 18. Using `py` on Windows

See installed Python versions:

```cmd
py -0
```

Run a specific version:

```cmd
py -3.11
```

Create an environment using Python 3.11:

```cmd
py -3.11 -m venv .venv
```

---

# 19. Upgrade pip

Inside the active environment:

```bash
python -m pip install --upgrade pip
```

Check the version:

```bash
python -m pip --version
```

---

# 20. Useful `pip` Commands

| Command | Purpose |
|---|---|
| `python -m pip list` | List installed packages |
| `python -m pip show numpy` | Show package information |
| `python -m pip install numpy` | Install package |
| `python -m pip install numpy==1.26.4` | Install specific version |
| `python -m pip install --upgrade numpy` | Upgrade package |
| `python -m pip uninstall numpy` | Remove package |
| `python -m pip freeze` | Show exact installed versions |
| `python -m pip freeze > requirements.txt` | Save dependencies |
| `python -m pip install -r requirements.txt` | Install saved dependencies |

---

# 21. Complete AI Project Workflow ⭐

This is the most important workflow for students to memorize.

### Step 1 — Create project

```bash
mkdir my_ai_project
cd my_ai_project
```

### Step 2 — Create environment

```bash
python -m venv .venv
```

### Step 3 — Activate

Windows:

```cmd
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Step 4 — Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 5 — Install AI packages

```bash
python -m pip install numpy pandas matplotlib
```

For a deep-learning project, for example:

```bash
python -m pip install torch torchvision
```

### Step 6 — Check packages

```bash
python -m pip list
```

### Step 7 — Save dependencies

```bash
python -m pip freeze > requirements.txt
```

### Step 8 — Work on your project

```bash
python main.py
```

### Step 9 — Finish work

```bash
deactivate
```

---

# 🧠 The 6 Commands Students Should Remember

If students forget everything else, remember these:

### 1. Create

```bash
python -m venv .venv
```

### 2. Activate — Windows

```bash
.venv\Scripts\activate
```

### 3. Activate — Linux/macOS

```bash
source .venv/bin/activate
```

### 4. Install

```bash
python -m pip install package_name
```

### 5. Save dependencies

```bash
python -m pip freeze > requirements.txt
```

### 6. Deactivate

```bash
deactivate
```

---

# 🔥 Remember the Concept

```text
              AI PROJECT
                  │
                  ▼
             Create .venv
                  │
                  ▼
              Activate
                  │
                  ▼
          Install Packages
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      NumPy     Pandas    PyTorch
        │         │         │
        └─────────┼─────────┘
                  ▼
             Write Code
                  │
                  ▼
        Save requirements.txt
                  │
                  ▼
              Deactivate
```

### One-line definition

> **A virtual environment is an isolated Python environment that allows each project to have its own packages and package versions without interfering with other projects.**