# Fecha de creación: 2026-10-03
"""EcoLima ML v1: verificación del paquete de resultados y cálculo de la propensión de emplazamiento.

La propensión mide cuánto se parece un sitio a los lugares donde ya existen puntos de reciclaje
(aprendizaje presencia–fondo). No es idoneidad, viabilidad ni probabilidad calibrada.
"""

from ecolima_ml.config import RUTA_PAQUETE, VERSIONES
from ecolima_ml.modelo import ModeloPropension, ResultadoPropension
from ecolima_ml.paquete import ErrorPaquete, Paquete

__version__ = "1.0.0"

__all__ = ["ErrorPaquete", "ModeloPropension", "Paquete", "ResultadoPropension", "RUTA_PAQUETE", "VERSIONES"]
