import os

import pandas as pd
import pyray as pr

from src.file_dialog import get_user_file
from src.nn_model import Model, get_data_dataframe, get_image
from src.ui import Button


class App:
    def __init__(self):
        self._btn_select_model = Button(
            pr.Rectangle(80, 50, 200, 50),
            "Choose NN model",
            pr.WHITE,
            pr.SKYBLUE,
            pr.GREEN,
        )
        self._btn_select_file = Button(
            pr.Rectangle(300, 50, 200, 50),
            "Choose data file",
            pr.WHITE,
            pr.SKYBLUE,
            pr.GREEN,
        )
        self._btn_run = Button(
            pr.Rectangle(80, 700, 150, 50),
            "Run model",
            pr.WHITE,
            pr.SKYBLUE,
            pr.GREEN,
        )
        self._model: Model | None = None
        self._data: pd.DataFrame | None = None
        pr.init_window(800, 800, "Gas Sensor GUI")
        self._img = pr.load_image("src/assets/theme.png")
        pr.image_resize(self._img, 640, 500)
        self._texture = pr.load_texture_from_image(self._img)
        self._img_rect = pr.Rectangle(80, 150, 640, 500)

        self._data_message: str | None = None
        self._data_message_color: pr.Color = pr.GREEN
        self._model_message: str | None = None
        self._model_message_color: pr.Color = pr.GREEN

    def load_model(self):
        model_path = get_user_file()
        try:
            self._model = Model(model_path)
        except Exception as e:
            print("Failed to open model file, ", e)
            self._model_message = "Error: Failed to load model."
            self._model_message_color = pr.RED
            return
        self._model_message = "File imported: " + os.path.basename(model_path)
        self._model_message_color = pr.GREEN
        print("Model path loaded: ", model_path)

    def load_data(self):
        data_path = get_user_file()
        if data_path is None:
            self._data_message = "Error: File is not selected!"
            self._data_message_color = pr.RED
            return
        self._data_message = "File imported: " + os.path.basename(data_path)
        self._data_message_color = pr.GREEN
        self._data = get_data_dataframe(data_path)
        print("Data path loaded: ", data_path)

    def update_img(self):
        if self._model is None or self._data is None:
            return

        time = self._data.iloc[:, 0].to_numpy()
        pred = self._model.predict(self._data)
        file_data = get_image(time, pred)
        self._img = pr.load_image_from_memory(
            ".png", file_data, len(file_data)
        )
        self._texture = pr.load_texture_from_image(self._img)

    def loop(self):
        if self._btn_select_model.is_clicked():
            self.load_model()

        if self._btn_select_file.is_clicked():
            self.load_data()

        if self._btn_run.is_clicked():
            self.update_img()

        pr.begin_drawing()
        pr.clear_background(pr.WHITE)

        self._btn_select_model.draw()
        if self._model_message is not None:
            pr.draw_text(
                self._model_message,
                int(self._btn_select_model._area.x),
                int(
                    self._btn_select_model._area.y
                    + self._btn_select_model._area.height * 1.1
                ),
                15,
                self._model_message_color,
            )
        self._btn_select_file.draw()
        if self._data_message is not None:
            pr.draw_text(
                self._data_message,
                int(self._btn_select_file._area.x),
                int(
                    self._btn_select_file._area.y
                    + self._btn_select_file._area.height * 1.1
                ),
                15,
                self._data_message_color,
            )

        if not (self._model is None or self._data is None):
            self._btn_run.draw()

        pr.draw_texture(self._texture, 80, 150, pr.WHITE)
        pr.draw_rectangle_lines_ex(self._img_rect, 3, pr.GRAY)

        pr.end_drawing()

    def run(self):
        pr.set_target_fps(60)
        while not pr.window_should_close():
            self.loop()


if __name__ == "__main__":
    app = App()
    app.run()
