# === Entorno Virtual y Dependencias ===
python -m venv venv --without-pip
.\venv\Scripts\Activate.ps1
python -m ensurepip
python -m pip install --upgrade pip
pip install -r requirements.txt

# === Pipeline de Ejecución de Redes Neuronales ===
python -m src.models.dataset
python -m src.services.train --epochs 10 --batch_size 32 --lr 0.001
python -m src.services.train --epochs 20 --batch_size 64 --lr 0.0001
python -m src.services.evaluate

# === Control de Versiones ===
git status
git add .
git commit -m "exp: baseline model training"
git push origin main