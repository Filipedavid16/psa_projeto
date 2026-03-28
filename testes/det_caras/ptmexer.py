import time
import serial
import serial.tools.list_ports


def listar_portas():
    return list(serial.tools.list_ports.comports())


def encontrar_porta_ptu(preferred_port=None, baudrate=38400, timeout=0.4):
    """
    Tenta encontrar automaticamente a porta do PTU.
    Prioridade:
    1) preferred_port, se existir
    2) portas cujo nome/descrição pareçam USB/Serial/COM
    """
    portas = listar_portas()

    if preferred_port:
        for p in portas:
            if p.device.upper() == preferred_port.upper():
                return p.device

    candidatos = []
    for p in portas:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        nome = (p.device or "").lower()

        score = 0
        if "usb" in desc or "usb" in hwid:
            score += 3
        if "serial" in desc or "uart" in desc:
            score += 2
        if "com" in nome:
            score += 1

        candidatos.append((score, p.device, p.description))

    candidatos.sort(reverse=True)

    for score, device, desc in candidatos:
        if score <= 0:
            continue
        try:
            ser = serial.Serial(device, baudrate, timeout=timeout)
            time.sleep(0.15)
            ser.close()
            return device
        except Exception:
            continue

    return None


class PTUController:
    def __init__(
        self,
        porta=None,
        baudrate=38400,
        timeout=0.5,
        pan_min=-5000,
        pan_max=5000,
        tilt_min=-500,
        tilt_max=500,
        pan_sign=1,
        tilt_sign=1,
        kp_pan=2400.0,
        kd_pan=1100.0,
        kp_tilt=1500.0,
        kd_tilt=550.0,
        deadzone_x=0.03,
        deadzone_y=0.02,
        max_step_pan=300,
        max_step_tilt=140,
        min_step_pan=12,
        min_step_tilt=8,
        cmd_interval=0.05,
        response_pause=0.01,
        auto_detect=True,
    ):
        self.porta = porta
        self.baudrate = baudrate
        self.timeout = timeout

        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max

        self.pan_sign = pan_sign
        self.tilt_sign = tilt_sign

        self.kp_pan = kp_pan
        self.kd_pan = kd_pan
        self.kp_tilt = kp_tilt
        self.kd_tilt = kd_tilt

        self.deadzone_x = deadzone_x
        self.deadzone_y = deadzone_y

        self.max_step_pan = max_step_pan
        self.max_step_tilt = max_step_tilt
        self.min_step_pan = min_step_pan
        self.min_step_tilt = min_step_tilt

        self.cmd_interval = cmd_interval
        self.response_pause = response_pause
        self.auto_detect = auto_detect

        self.ser = None

        # posição relativa mantida pelo programa
        self.pan_atual = 0
        self.tilt_atual = 0

        # tracking
        self.last_t = None
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.last_cmd_t = 0.0

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def ligar(self):
        if self.ser is not None:
            return

        porta_final = self.porta
        if not porta_final and self.auto_detect:
            porta_final = encontrar_porta_ptu(baudrate=self.baudrate, timeout=self.timeout)

        if not porta_final:
            raise RuntimeError("Não foi possível detetar automaticamente a porta do PTU.")

        self.porta = porta_final
        self.ser = serial.Serial(self.porta, self.baudrate, timeout=self.timeout)
        time.sleep(0.2)
        print(f"PTU ligado em {self.porta}")

    def fechar(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _enviar(self, cmd):
        if self.ser is None:
            return ""

        self.ser.write((cmd + "\r").encode("ascii"))
        self.ser.flush()
        time.sleep(self.response_pause)

        try:
            resp = self.ser.read_all().decode(errors="ignore").strip()
        except Exception:
            resp = ""

        if resp:
            print("<", resp)
        return resp

    def mover_pan(self, delta):
        if delta == 0:
            return

        novo = self.pan_atual + delta
        novo = self._clamp(novo, self.pan_min, self.pan_max)
        delta_real = novo - self.pan_atual

        if delta_real == 0:
            return

        self._enviar(f"PO{int(delta_real)}")
        self.pan_atual = novo
        print(f"PAN -> {self.pan_atual}")

    def mover_tilt(self, delta):
        if delta == 0:
            return

        novo = self.tilt_atual + delta
        novo = self._clamp(novo, self.tilt_min, self.tilt_max)
        delta_real = novo - self.tilt_atual

        if delta_real == 0:
            return

        self._enviar(f"TO{int(delta_real)}")
        self.tilt_atual = novo
        print(f"TILT -> {self.tilt_atual}")

    def voltar_origem(self):
        """
        Volta ao zero relativo do programa.
        """
        try:
            if self.pan_atual != 0:
                self.mover_pan(-self.pan_atual)
            if self.tilt_atual != 0:
                self.mover_tilt(-self.tilt_atual)
        except Exception as e:
            print("Erro ao voltar à origem:", e)

    def reset_tracking(self):
        self.last_t = None
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0

    def track_face(self, bbox, frame_w, frame_h):
        if bbox is None:
            self.reset_tracking()
            return

        now = time.time()
        if now - self.last_cmd_t < self.cmd_interval:
            return

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        center_x = frame_w / 2.0
        center_y = frame_h / 2.0

        err_x = (cx - center_x) / max(frame_w, 1)
        err_y = (cy - center_y) / max(frame_h, 1)

        if self.last_t is None:
            derr_x = 0.0
            derr_y = 0.0
        else:
            dt = max(now - self.last_t, 1e-3)
            derr_x = (err_x - self.prev_err_x) / dt
            derr_y = (err_y - self.prev_err_y) / dt

        self.last_t = now
        self.prev_err_x = err_x
        self.prev_err_y = err_y

        if abs(err_x) < self.deadzone_x:
            err_x = 0.0
            derr_x = 0.0

        if abs(err_y) < self.deadzone_y:
            err_y = 0.0
            derr_y = 0.0

        cmd_pan = int(self.pan_sign * (self.kp_pan * err_x + self.kd_pan * derr_x))
        cmd_tilt = int(self.tilt_sign * (self.kp_tilt * err_y + self.kd_tilt * derr_y))

        if cmd_pan > 0:
            cmd_pan = max(cmd_pan, self.min_step_pan)
        elif cmd_pan < 0:
            cmd_pan = min(cmd_pan, -self.min_step_pan)

        if cmd_tilt > 0:
            cmd_tilt = max(cmd_tilt, self.min_step_tilt)
        elif cmd_tilt < 0:
            cmd_tilt = min(cmd_tilt, -self.min_step_tilt)

        cmd_pan = self._clamp(cmd_pan, -self.max_step_pan, self.max_step_pan)
        cmd_tilt = self._clamp(cmd_tilt, -self.max_step_tilt, self.max_step_tilt)

        if cmd_pan != 0:
            self.mover_pan(cmd_pan)
        if cmd_tilt != 0:
            self.mover_tilt(cmd_tilt)

        self.last_cmd_t = now