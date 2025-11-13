# Quickstart Commands

```bash
# 1. Install dependencies (optional virtualenv recommended)
python3 -m pip install -r requirements.txt

# 2. Run provided sample fixtures for both tools
python3 run_samples.py "python3 factory/main.py" "python3 belts/main.py"

# 3. Execute the automated test suite
FACTORY_CMD="python3 factory/main.py" BELTS_CMD="python3 belts/main.py" python3 -m pytest -q
```