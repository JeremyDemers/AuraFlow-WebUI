@echo off

python -m venv venv
call venv/scripts/activate

@echo Installing PyTorch for python 3.11.x or 3.12.X and CUDA 12.4

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

@echo Installing requirements...

pip install -r requirements.txt

@echo Install complete 🎉  Press any key to close this screen...
pause