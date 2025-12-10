import pyray as pr


class Button:
    def __init__(self, rect, text, def_col, tog_col, set_col):
        self._area = rect
        self._text = text
        self._default_color = def_col
        self._toggled_color = tog_col
        self._set_color = set_col
        self._cur_color = def_col

    def draw(self):
        pr.draw_rectangle_rec(self._area, self._cur_color)
        pr.draw_rectangle_lines_ex(self._area, 3, pr.GRAY)
        text_x = int(self._area.x + 0.3 * self._area.height)
        text_y = int(self._area.y + 0.3 * self._area.height)
        font_size = int(0.4 * self._area.height)
        pr.draw_text(self._text, text_x, text_y, font_size, pr.GRAY)

    def is_clicked(self):
        is_focused = pr.check_collision_point_rec(
            pr.get_mouse_position(), self._area
        )
        is_released = pr.is_mouse_button_released(pr.MOUSE_LEFT_BUTTON)

        if is_focused:
            self._cur_color = self._toggled_color
        else:
            self._cur_color = self._default_color

        return is_focused and is_released
