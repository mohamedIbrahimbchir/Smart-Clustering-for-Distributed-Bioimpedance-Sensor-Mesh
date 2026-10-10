════════════════════════════════════════════════════════════════════
  PROJECT 7 — Smart Clustering for Distributed Bioimpedance Mesh
  Sensors & Smart IoT — Technische Universität Chemnitz
  Supervisor: Ahmed Yahia Kallel
════════════════════════════════════════════════════════════════════

  README — Setup Guide for Team Members
  Read this file carefully before doing anything.

════════════════════════════════════════════════════════════════════
  SECTION 1 — WHAT IS THIS PROJECT?
════════════════════════════════════════════════════════════════════

This project simulates a wireless sensor network of 20 bioimpedance
patches placed on a patient's body. Each patch measures electrical
properties of tissue (bioimpedance). The patches communicate via
Bluetooth Low Energy (BLE) and must deliver data to a central gateway
while running on tiny coin-cell batteries for 8 hours.

We wrote Python algorithms that:
  - Organise nodes into clusters (clustering algorithms)
  - Decide when each node sleeps or stays awake (sleep schedulers)

Everything runs inside a Python simulator — NO hardware needed.

════════════════════════════════════════════════════════════════════
  SECTION 2 — REQUIREMENTS
════════════════════════════════════════════════════════════════════

  ⚠ IMPORTANT: Python version MUST be 3.10 or higher.
               We recommend Python 3.11, 3.12, or 3.13.
               Python 3.9 or older will NOT work (syntax error).

  How to check your Python version:
    Open a terminal / command prompt and type:
      python --version

  If your version is below 3.10, download the latest Python from:
    https://www.python.org/downloads/

  ⚠ During installation on Windows:
    Check the box "Add python.exe to PATH" before clicking Install.
    This is VERY important — without it, nothing will work.

════════════════════════════════════════════════════════════════════
  SECTION 3 — HOW TO OPEN THE PROJECT IN VS CODE
════════════════════════════════════════════════════════════════════

  1. Open Visual Studio Code.
  2. Click File → Open Folder.
  3. Select the project folder (Project_7).
  4. Open the integrated terminal:
       View → Terminal   (or press Ctrl + `)
  5. Make sure the terminal shows the project folder path, e.g.:
       PS C:\Users\YourName\Desktop\Project_7>

  ⚠ All commands in this guide must be run from this folder.
    Do NOT run them from inside src/ or student_work/.

════════════════════════════════════════════════════════════════════
  SECTION 4 — INSTALL DEPENDENCIES
════════════════════════════════════════════════════════════════════

  The project uses NumPy, SciPy, Matplotlib, and NetworkX.
  All required packages are listed in requirements.txt.

  Run this ONE command to install everything:

    pip install -r requirements.txt

  Wait for it to finish. You should see "Successfully installed..."
  at the end.

  ⚠ If you see "pip is not recognised", try:
      python -m pip install -r requirements.txt

  ⚠ If you are using a virtual environment (.venv), make sure it
    is activated first:
      Windows:   .venv\Scripts\activate
      Mac/Linux: source .venv/bin/activate

════════════════════════════════════════════════════════════════════
  SECTION 5 — PROJECT FOLDER STRUCTURE
════════════════════════════════════════════════════════════════════

  Project_7_Distributed_Bioimpedance/
  │
  ├── auto_grader.py              ← Run this to check our grade
  ├── experiments.py              ← Run this to see demo simulations
  ├── requirements.txt            ← List of Python packages needed
  ├── README.md                   ← Original project description
  │
  ├── src/                        ← Simulator core (DO NOT MODIFY)
  │   ├── config.py               ← All simulation parameters
  │   ├── node.py                 ← Sensor node model
  │   ├── rssi_model.py           ← BLE signal propagation model
  │   ├── clustering.py           ← 4 built-in clustering algorithms
  │   ├── sleep_scheduler.py      ← 4 built-in sleep schedulers
  │   ├── mesh_network.py         ← Multi-hop routing
  │   ├── energy_model.py         ← Battery consumption model
  │   ├── metrics.py              ← PDR, balance, efficiency metrics
  │   └── simulator.py            ← Main simulation orchestrator
  │
  └── student_work/               ← OUR CODE IS HERE
      ├── my_clustering.py        ← Our 2 clustering algorithms ✓
      ├── my_sleep_scheduler.py   ← Our 2 sleep schedulers ✓
      ├── evaluate_my_work.py     ← Compare our algos vs baselines
      └── my_experiments.py       ← Custom experiment sandbox

  ⚠ Only files inside student_work/ were modified.
    Do NOT modify anything inside src/.

════════════════════════════════════════════════════════════════════
  SECTION 6 — HOW TO RUN THE AUTO-GRADER
════════════════════════════════════════════════════════════════════

  The auto-grader checks all our algorithms and gives a score.
  Run it with:

    python auto_grader.py

  Expected output — FINAL VERDICT should say PASSED:

    ✓ student_clustering_1    75/100   PASSED
    ✓ student_clustering_2    78/100   PASSED
    ✓ student_scheduler_1     81/100   PASSED
    ✓ student_scheduler_2     71/100   PASSED
    ✓ FINAL VERDICT:  PASSED

  ⚠ The grader takes 1-3 minutes to run. This is normal.
    Do not close the terminal while it is running.

════════════════════════════════════════════════════════════════════
  SECTION 7 — HOW TO RUN THE DEMO EXPERIMENTS
════════════════════════════════════════════════════════════════════

  The experiments show visual simulations with plots and charts.
  Run with:

    python experiments.py

  You will see a menu like this:

    [1] Network topology visualisation
    [2] Compare clustering algorithms
    [3] Compare sleep schedulers
    [4] Node scaling study
    [5] RSSI threshold sweep
    [6] Full dashboard (1 hour)
    [7] Wearable bioimpedance challenge
    [0] Run ALL experiments

  Type a number and press Enter to run that experiment.
  Charts will open in a separate window — close them to continue.

  RECOMMENDED experiments to run for the presentation:
    → Experiment 2: shows our clustering vs built-in baselines
    → Experiment 3: shows our schedulers vs built-in baselines
    → Experiment 6: full dashboard with energy and PDR plots
    → Experiment 7: the wearable challenge (20 nodes, 8 hours)

  ⚠ Experiment 7 simulates 8 hours and may take 2-5 minutes.

════════════════════════════════════════════════════════════════════
  SECTION 8 — HOW TO RUN THE EVALUATION (OUR ALGOS VS BASELINES)
════════════════════════════════════════════════════════════════════

  To compare our algorithms against the 4 built-in baseline
  algorithms, run:

    python student_work/evaluate_my_work.py

  This prints a comparison table showing PDR, cluster balance,
  energy efficiency, and network lifetime for each algorithm.


════════════════════════════════════════════════════════════════════
  SECTION 9 — COMMON ERRORS AND FIXES
════════════════════════════════════════════════════════════════════

  ERROR: "ModuleNotFoundError: No module named 'numpy'"
  FIX:   Run: pip install -r requirements.txt

  ERROR: "TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'"
  FIX:   Your Python is older than 3.10. Install Python 3.11+
         from https://www.python.org/downloads/

  ERROR: "python is not recognized as an internal or external command"
  FIX:   Python is not in your PATH. Reinstall Python and check
         the "Add python.exe to PATH" box during installation.

  ERROR: Plots don't appear / window freezes
  FIX:   Close the plot window to continue to the next experiment.
         If no window appears, run: pip install matplotlib --upgrade

  ERROR: auto_grader.py takes forever
  FIX:   This is normal. The 8-hour simulation in T10 takes time.
         Wait 2-4 minutes for it to complete.

════════════════════════════════════════════════════════════════════
  SECTION 11 — QUICK START (3 steps)
════════════════════════════════════════════════════════════════════

  Step 1 — Install packages:
    pip install -r requirements.txt

  Step 2 — Check our grade:
    python auto_grader.py

  Step 3 — See the demo experiments:
    python experiments.py
step 4 : python student_work/evaluate_my_work.py


