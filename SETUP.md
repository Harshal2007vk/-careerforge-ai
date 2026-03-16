# CareerForge-AI — Setup Guide (Start Here)

**Never set this up before? Do this:**

1. Install **Python** from https://www.python.org/downloads/ (check **“Add Python to PATH”**).
2. Double-click **`setup.bat`** in this folder (installs packages and trains the model).
3. Double-click **`run_app.bat`** to open the app in your browser.

If that works, you’re done. If not, follow the full steps below.

---

Follow these steps **in order**. You only need to do Steps 1–2 once. After that, use Step 3 to train and Step 4 to run the app.

---

## Step 1: Install Python (if you don’t have it)

1. **Check if Python is installed**
   - Press `Win + R`, type `cmd`, press Enter.
   - In the black window, type:  
     `python --version`  
   - If you see something like `Python 3.11` or `Python 3.12`, you’re good → go to **Step 2**.
   - If you see “not recognized” or an error, install Python:

2. **Install Python**
   - Go to: https://www.python.org/downloads/
   - Click **“Download Python 3.x.x”**.
   - Run the installer.
   - **Important:** On the first screen, **check the box “Add Python to PATH”**.
   - Click **“Install Now”** and finish.
   - Close and reopen the terminal (or restart the computer), then run `python --version` again to confirm.

---

## Step 2: Open the project folder in the terminal

1. Open **File Explorer** and go to:  
   `C:\Users\A15\Desktop\CareerForge-AI`
2. **Click the address bar** at the top (where it shows the path), type `cmd`, and press **Enter**.  
   A black window (Command Prompt) will open **already in your project folder**.

   **Or:**  
   - Press `Win + R`, type `cmd`, Enter.  
   - Then type:  
     `cd C:\Users\A15\Desktop\CareerForge-AI`  
   and press Enter.

You should see a line ending with `...\CareerForge-AI>`.

---

## Step 3: Install the required packages

In the same terminal, run:

```text
pip install -r requirements.txt
```

Wait until it finishes (no red errors). If you see “Successfully installed …”, you’re done.

---

## Step 4: Train the model (first time and after data changes)

In the same terminal, run **one** of these:

**Option A — For the web app (simplest):**

```text
python train_model.py
```

- Creates `career_model.pkl`. The Streamlit app uses this.
- When you see “Recommender artifacts saved to career_model.pkl”, you’re done.

**Option B — Full ML pipeline (all models, for later use):**

```text
python train_pipeline.py
```

- Takes longer. Saves everything under `ml_artifacts/`.
- Use this if you want to use the full ML pipeline (see README).
- Quick test with less data: `python train_pipeline.py 0.1`

---

## Step 5: Run the web app

In the same terminal, run:

```text
streamlit run app.py
```

- A browser tab should open automatically with the app.
- If not, the terminal will show a link like `http://localhost:8501` — click it or copy it into your browser.

You can now use the app: enter your profile (age, CGPA, scores, etc.) and click **“Analyze my profile”** to get career recommendations.

To **stop the app**: go back to the terminal and press `Ctrl + C`.

---

## Easiest way: double-click (Windows)

1. **First time only:** Double-click `setup.bat`  
   - It installs packages and runs `train_model.py` so the app has a model.
2. **Every time you want to use the app:** Double-click `run_app.bat`  
   - The app opens in your browser at http://localhost:8501

---

## Summary — what to type and when

| When              | What to type                       |
|-------------------|-------------------------------------|
| One-time setup    | `pip install -r requirements.txt`  |
| Train for app     | `python train_model.py`             |
| To use the app    | `streamlit run app.py`              |

---

## If something goes wrong

- **“python is not recognized”**  
  Install Python (Step 1) and make sure you checked **“Add Python to PATH”**. Then close and reopen the terminal.

- **“pip is not recognized”**  
  Try: `python -m pip install -r requirements.txt`

- **Error when running train_pipeline.py**  
  Make sure you’re in the project folder (Step 2) and that the file `career_guidance_perfect.csv` is in the same folder.

- **App doesn’t open in browser**  
  Manually open a browser and go to: `http://localhost:8501`

If you still get an error, copy the **full error message** from the terminal and share it so someone can help you fix it.
