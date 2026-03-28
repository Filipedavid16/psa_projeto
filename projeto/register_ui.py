import cv2
import numpy as np

from config import REGISTER_WINDOW, PENDENTES_DIR


class RegisterUI:
    def __init__(self):
        self.is_open = False

        self.selected_pending_id = None
        self.first_name = ""
        self.last_name = ""

        self.active_field = None
        self.dropdown_open = False
        self.dropdown_items = []

        # Retângulos da interface
        self.combo_rect = (60, 95, 330, 145)
        self.arrow_rect = (330, 95, 380, 145)

        self.photo_rect = (60, 185, 360, 465)

        self.first_name_rect = (450, 170, 780, 220)
        self.last_name_rect = (450, 280, 780, 330)

        self.delete_btn_rect = (450, 395, 610, 450)
        self.register_btn_rect = (630, 395, 780, 450)

    def toggle(self):
        self.is_open = not self.is_open

        if not self.is_open:
            self.selected_pending_id = None
            self.first_name = ""
            self.last_name = ""
            self.active_field = None
            self.dropdown_open = False

            try:
                cv2.destroyWindow(REGISTER_WINDOW)
            except cv2.error:
                pass

    def _point_in_rect(self, x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _draw_textbox(self, canvas, rect, text, active=False):
        x1, y1, x2, y2 = rect
        border_color = (0, 180, 255) if active else (255, 255, 255)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2)
        cv2.putText(
            canvas,
            text,
            (x1 + 10, y1 + 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    def _draw_button(self, canvas, rect, text, bg_color, text_color=(255, 255, 255)):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2

        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_color,
            2
        )

    def _get_preview_image(self, track_id):
        if track_id is None:
            return None

        folder = PENDENTES_DIR / f"ID_{track_id}"
        if not folder.exists():
            return None

        photos = sorted(folder.glob("*.jpg"))
        if not photos:
            return None

        image = cv2.imread(str(photos[0]))
        return image

    def _draw_preview(self, canvas):
        x1, y1, x2, y2 = self.photo_rect

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        image = self._get_preview_image(self.selected_pending_id)
        if image is None:
            cv2.putText(
                canvas,
                "Sem foto",
                (x1 + 85, y1 + 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )
            return

        ih, iw = image.shape[:2]
        box_w = x2 - x1 - 20
        box_h = y2 - y1 - 20

        scale = min(box_w / iw, box_h / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))

        resized = cv2.resize(image, (new_w, new_h))

        px = x1 + (x2 - x1 - new_w) // 2
        py = y1 + (y2 - y1 - new_h) // 2

        canvas[py:py + new_h, px:px + new_w] = resized

    def _draw_combobox(self, canvas, pending_ids):
        x1, y1, x2, y2 = self.combo_rect
        ax1, ay1, ax2, ay2 = self.arrow_rect

        cv2.putText(
            canvas,
            "IDs pendentes",
            (60, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), (255, 255, 255), 2)

        selected_text = f"ID {self.selected_pending_id}" if self.selected_pending_id is not None else ""
        cv2.putText(
            canvas,
            selected_text,
            (x1 + 12, y1 + 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # seta
        cx = (ax1 + ax2) // 2
        cy = (ay1 + ay2) // 2
        triangle = np.array([
            [cx - 10, cy - 5],
            [cx + 10, cy - 5],
            [cx, cy + 8]
        ], np.int32)
        cv2.fillPoly(canvas, [triangle], (255, 255, 255))

        self.dropdown_items = []

        if self.dropdown_open:
            item_h = 40
            start_y = y2

            for i, track_id in enumerate(pending_ids):
                iy1 = start_y + i * item_h
                iy2 = iy1 + item_h
                rect = (x1, iy1, ax2, iy2)

                cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (45, 45, 45), -1)
                cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 1)

                cv2.putText(
                    canvas,
                    f"ID {track_id}",
                    (rect[0] + 12, rect[1] + 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                self.dropdown_items.append((rect, track_id))

    def draw(self, registry):
        width = 860
        height = 520
        canvas = np.full((height, width, 3), 25, dtype=np.uint8)

        cv2.rectangle(canvas, (20, 20), (840, 500), (255, 255, 255), 2)

        cv2.putText(
            canvas,
            "Janela de Registo - ID's",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            2
        )

        pending_ids = registry.get_pending_ids()

        if self.selected_pending_id is not None and self.selected_pending_id not in pending_ids:
            self.selected_pending_id = None
            self.first_name = ""
            self.last_name = ""
            self.dropdown_open = False

        self._draw_combobox(canvas, pending_ids)
        self._draw_preview(canvas)

        cv2.putText(
            canvas,
            "Nome",
            (450, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        self._draw_textbox(
            canvas,
            self.first_name_rect,
            self.first_name,
            active=(self.active_field == "first_name")
        )

        cv2.putText(
            canvas,
            "Apelido",
            (450, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        self._draw_textbox(
            canvas,
            self.last_name_rect,
            self.last_name,
            active=(self.active_field == "last_name")
        )

        self._draw_button(canvas, self.delete_btn_rect, "Eliminar registo", (60, 60, 160))
        self._draw_button(canvas, self.register_btn_rect, "Registar", (60, 140, 60))

        cv2.imshow(REGISTER_WINDOW, canvas)

    def _register_selected(self, registry, tracks):
        if self.selected_pending_id is None:
            print("Seleciona primeiro um ID.")
            return

        full_name = f"{self.first_name.strip()} {self.last_name.strip()}".strip()
        if not full_name:
            print("Preenche o nome e/ou apelido.")
            return

        ok = registry.register_pending_id(self.selected_pending_id, full_name, tracks)
        if ok:
            self.selected_pending_id = None
            self.first_name = ""
            self.last_name = ""
            self.active_field = None
            self.dropdown_open = False

    def _delete_selected(self, registry, tracks):
        if self.selected_pending_id is None:
            print("Seleciona primeiro um ID.")
            return

        ok = registry.delete_pending_id(self.selected_pending_id, tracks)
        if ok:
            self.selected_pending_id = None
            self.first_name = ""
            self.last_name = ""
            self.active_field = None
            self.dropdown_open = False

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        registry = param["registry"] if param else None
        tracks = param["tracks"] if param else None

        # Combobox
        if self._point_in_rect(x, y, self.combo_rect) or self._point_in_rect(x, y, self.arrow_rect):
            self.dropdown_open = not self.dropdown_open
            self.active_field = None
            return

        # Itens da dropdown
        if self.dropdown_open:
            clicked_item = False
            for rect, track_id in self.dropdown_items:
                if self._point_in_rect(x, y, rect):
                    self.selected_pending_id = track_id
                    self.dropdown_open = False
                    clicked_item = True
                    break

            if clicked_item:
                return

            self.dropdown_open = False

        # Textboxes
        if self._point_in_rect(x, y, self.first_name_rect):
            self.active_field = "first_name"
            return

        if self._point_in_rect(x, y, self.last_name_rect):
            self.active_field = "last_name"
            return

        # Botões
        if self._point_in_rect(x, y, self.delete_btn_rect):
            self._delete_selected(registry, tracks)
            return

        if self._point_in_rect(x, y, self.register_btn_rect):
            self._register_selected(registry, tracks)
            return

        self.active_field = None

    def process_key(self, key, registry, tracks):
        if key == 255:
            return

        if key == 13:  # Enter
            self._register_selected(registry, tracks)
            return

        if key == 27:  # Esc
            self.active_field = None
            self.dropdown_open = False
            return

        if key == 9:  # Tab
            if self.active_field == "first_name":
                self.active_field = "last_name"
            else:
                self.active_field = "first_name"
            return

        if key in (8, 127):  # Backspace
            if self.active_field == "first_name":
                self.first_name = self.first_name[:-1]
            elif self.active_field == "last_name":
                self.last_name = self.last_name[:-1]
            return

        if 32 <= key <= 126:
            if self.active_field == "first_name":
                self.first_name += chr(key)
            elif self.active_field == "last_name":
                self.last_name += chr(key)