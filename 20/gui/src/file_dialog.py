import tkinter as tk
from pathlib import Path
from tkinter import filedialog as fd


def get_user_file(init_dir: Path | None = None) -> Path | None:
    if init_dir is None:
        init_dir = Path.home()
    root = tk.Tk()
    root.withdraw()
    file_name = fd.askopenfilename(initialdir=init_dir)
    if file_name:
        return Path(file_name)
    return None
