# Quickstart Commands

```bash
# 1. Install dependencies (optional virtualenv recommended)
# Install required dependencies based on the need. 

# 2. Run provided sample fixtures for both tools
python run_samples.py "python factory/main.py" "python belts/main.py"

# 3. Execute the automated test suite
FACTORY_CMD="python factory/main.py" BELTS_CMD="python belts/main.py" pytest -q
```