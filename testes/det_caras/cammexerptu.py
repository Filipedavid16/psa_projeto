import sys
import det_track_id_rot_ft_nom_rec
from ptmexer import PTUController

try:
    from ptmexer import encontrar_porta_ptu
except ImportError:
    encontrar_porta_ptu = None


def obter_porta_ptu():
    """
    Tenta descobrir automaticamente a porta do PTU.
    Se não existir a função encontrar_porta_ptu no ptmexer.py,
    pede ao utilizador uma porta manual.
    """
    if encontrar_porta_ptu is not None:
        porta = encontrar_porta_ptu()
        if porta:
            print(f"PTU encontrado em: {porta}")
            return porta

    print("Não foi possível detetar automaticamente a porta do PTU.")
    porta = input("Indica a porta manualmente (ex: COM15): ").strip()
    return porta if porta else None


def main():
    porta = obter_porta_ptu()
    if not porta:
        print("Sem porta PTU. A terminar.")
        sys.exit(1)

    ptu = None

    try:
        ptu = PTUController(
            porta=porta,
            baudrate=38400,
            timeout=0.5,

            pan_min=-5000,
            pan_max=5000,
            tilt_min=-500,
            tilt_max=500,

            pan_sign=-1,   # ajusta se pan estiver invertido
            tilt_sign=1,   # ajusta se tilt estiver invertido

            kp_pan=2400.0,
            kd_pan=1100.0,
            kp_tilt=1500.0,
            kd_tilt=550.0,

            deadzone_x=0.03,
            deadzone_y=0.00,

            max_step_pan=300,
            max_step_tilt=140,

            min_step_pan=12,
            min_step_tilt=8,

            cmd_interval=0.05,
            response_pause=0.01,
        )

        ptu.ligar()
        print("PTU ligado com sucesso.")

        # Se o teu psainicio.py estiver preparado para usar isto,
        # ele pode chamar psainicio.seguir_face(...)
        def seguir_face(bbox, frame_w, frame_h):
            if bbox is not None:
                ptu.track_face(bbox, frame_w, frame_h)
            else:
                ptu.reset_tracking()

        # expõe a função ao módulo psainicio
        psainicio.seguir_face = seguir_face

        # arranca o sistema de visão
        psainicio.run()

    except KeyboardInterrupt:
        print("Interrompido pelo utilizador.")

    except Exception as e:
        print("Erro no main.py:", e)

    finally:
        if ptu is not None:
            try:
                print("A voltar PTU à origem...")
                ptu.voltar_origem()
            except Exception as e:
                print("Erro ao voltar à origem:", e)

            try:
                ptu.fechar()
            except Exception:
                pass

        print("Programa terminado.")


if __name__ == "__main__":
    main()