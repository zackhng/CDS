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
CDS/
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

Here is a **cleaner README section in Markdown** with the adjustments you requested:

* Includes **creating `raw_data/` since it is in `.gitignore`**
* **Does not include the script code**
* Clearly states that the command must be run **from the project root directory**

You can paste this directly into your `README.md`.

---

## Dataset Preparation

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset for audio from Kaggle.

Because the dataset is large, the `raw_data/` directory is excluded from version control via `.gitignore`. Each user must download and prepare the dataset locally.

---
## Audio dataset

### 1. Create the Data Directory

From the **project root directory**, create the `raw_data` folder:

```bash
mkdir raw_data
```

Project structure should now look like:

```
CDS/
├── notebooks/
├── src/
├── raw_data/
└── README.md
```
Next, create a folder to store the processed audio files:
mkdir raw_data/ravdess_audio
This folder will be used to collect all .wav files in a single directory for easier data processing.
Updated structure:
```
CDS/
├── notebooks/
├── src/
├── raw_data/          
│   └── ravdess_audio/    # flattened audio files for processing
└── README.md
```

---

### 2. Download the Dataset

1. Download the **RAVDESS dataset** from Kaggle. Link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
2. Extract the downloaded archive.
3. Move the extracted dataset into the `raw_data` folder.

Expected structure:

```
CDS/
├── raw_data/
│   └── archive/
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ...
```

Each `Actor_xx` directory contains multiple `.wav` audio files.

---

### 3. Prepare the Dataset

The original dataset stores audio files inside multiple actor folders.
For easier processing and machine learning pipelines, all `.wav` files are collected into a single directory.

Run the dataset preparation script:

```bash
python src/data/audio/collect_audio_files.py
```

**Important:**
This command must be executed **from the project root directory**.

---

### 4. Resulting Dataset Structure

After running the script, the dataset will be organized as follows:

```
CDS/
├── raw_data/
│   ├── archive/        # original dataset
│   └── ravdess_audio/  # flattened dataset used for analysis
│       ├── 03-01-01-01-01-01-01.wav
│       ├── 03-01-01-01-01-01-02.wav
│       └── ...
```

The `ravdess_audio` folder contains all audio files in a single directory, which simplifies data loading, preprocessing, and feature extraction for machine learning tasks.

---

### 5. Verify the Dataset

You can verify the number of audio files using:

```bash
ls raw_data/ravdess_audio | wc -l
```

Expected output:

```
1440
```
---

### 6. Creating Metadata CSV (`metadata.csv`)

The `metadata.csv` contains labels extracted from the RAVDESS filenames. It includes columns such as:

* `file` – audio filename
* `emotion` – mapped from filename number
* `intensity` – normal / strong
* `statement` – “Kids are talking by the door” or “Dogs are sitting by the door”
* `gender` – male/female

### **How to generate it**

From the **project root directory**, run:

```bash
python src/data/audio/build_metadata.py
```

* This script parses all filenames in `raw_data/ravdess_audio/` and creates `raw_data/metadata.csv`.

### **Verify the CSV**

After running the script, inspect the first few rows:

```bash
head raw_data/metadata.csv
```

Example output:

file,emotion,intensity,statement,gender<br>
03-01-06-01-02-02-02.wav,fearful,normal,Dogs are sitting by the door,female<br>
03-01-05-01-02-01-16.wav,angry,normal,Dogs are sitting by the door,female<br>
03-01-08-01-01-01-14.wav,surprised,normal,Kids are talking by the door,female<br>
03-01-06-01-02-02-16.wav,fearful,normal,Dogs are sitting by the door,female<br>
03-01-05-01-02-01-02.wav,angry,normal,Dogs are sitting by the door,female<br>
03-01-01-01-02-02-06.wav,neutral,normal,Dogs are sitting by the door,female<br>
03-01-02-01-02-01-12.wav,calm,normal,Dogs are sitting by the door,female<br>
03-01-01-01-02-02-12.wav,neutral,normal,Dogs are sitting by the door,female<br>
03-01-02-01-02-01-06.wav,calm,normal,Dogs are sitting by the door,female<br>

---

After **flattening the audio files** and **generating `metadata.csv`**, the project structure looks like this:

```text id="v79oky"
CDS/
├── raw_data/
│   ├── archive/          # original dataset from Kaggle
│   ├── ravdess_audio/    # all 1440 audio files in a single folder
│   └── metadata.csv      # generated CSV with metadata
├── src/
│   └── data/audio/
│       ├── collect_audio_files.py  # script to flatten audio dataset
│       └── build_metadata.py       # script to generate metadata.csv
├── notebooks/                 
└── README.md
```

---

