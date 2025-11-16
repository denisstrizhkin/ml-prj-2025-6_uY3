import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from typing import Dict, List
import os


class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.total_losses = []
        self.data_losses = []
        self.pde_losses = []

    def add_epoch_data(self, epoch: int, total_loss: float, data_loss: float, pde_loss: float):
        """Добавление данных эпохи для построения графика"""
        self.epochs.append(epoch)
        self.total_losses.append(total_loss)
        self.data_losses.append(data_loss)
        self.pde_losses.append(pde_loss)

    def create_training_plot(self) -> str:
        """Создание графика обучения в base64 формате"""
        if not self.epochs:
            return None

        plt.figure(figsize=(12, 8))

        # Полупрозрачная область для общих потерь
        plt.fill_between(self.epochs, self.total_losses, alpha=0.3, color='blue', label='Общие потери')
        plt.plot(self.epochs, self.total_losses, 'b-', linewidth=2, label='Общие потери')

        # Потери данных
        plt.plot(self.epochs, self.data_losses, 'r--', linewidth=1.5, label='Потери данных (MSE)')

        # Потери PDE
        plt.plot(self.epochs, self.pde_losses, 'g--', linewidth=1.5, label='Потери уравнения (PDE)')

        plt.yscale('log')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери (log scale)')
        plt.title('График обучения PINN модели')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Сохраняем в base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def get_training_stats(self) -> Dict:
        """Получение статистики обучения"""
        if not self.epochs:
            return {}

        return {
            "final_epoch": self.epochs[-1],
            "final_total_loss": float(self.total_losses[-1]),
            "final_data_loss": float(self.data_losses[-1]),
            "final_pde_loss": float(self.pde_losses[-1]),
            "min_total_loss": float(min(self.total_losses)),
            "min_data_loss": float(min(self.data_losses)),
            "min_pde_loss": float(min(self.pde_losses))
        }