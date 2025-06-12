import torch
import torch.nn as nn
import os


def save_model(model: nn.Module, path: str, filename: str = "model.pth"):
    """Guarda el estado del modelo."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Modelo guardado en: {os.path.join(path, filename)}")


def load_model(model: nn.Module, path: str, filename: str = "model.pth"):
    """Carga el estado del modelo."""
    model_path = os.path.join(path, filename)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Modelo cargado desde: {model_path}")
    else:
        print(
            f"No se encontró el modelo en: {model_path}. Iniciando con pesos aleatorios.")


def setup_logging():
    """Configura el logging para el entrenamiento."""
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)
