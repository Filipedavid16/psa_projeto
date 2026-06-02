import sys
import psainicio
from ptmexer_waveshare import PTUController, encontrar_porta_ptu


def obter_porta_ptu():
    porta = encontrar_porta_ptu()

    if porta:
        print(f"PTU encontrado em: {porta}")
        return porta

    print("Não foi possível detetar automaticamente a porta do PTU.")
    porta = input("Indica a porta manualmente, exemplo COM18: ").strip()

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
            baudrate=115200,
            timeout=0.5,

            pan_min=-180,
            pan_max=180,
            tilt_min=-30,
            tilt_max=90,

            pan_sign=1,
            tilt_sign=-1,

            kp_pan=35.0,
            kd_pan=12.0,
            kp_tilt=25.0,
            kd_tilt=8.0,

            deadzone_x=0.04,
            deadzone_y=0.04,

            max_step_pan=4.0,
            max_step_tilt=3.0,

            min_step_pan=0.5,
            min_step_tilt=0.5,

            cmd_interval=0.08,
            response_pause=0.03,

            speed=60,
            acc=0,
        )

        ptu.ligar()
        print("PTU Waveshare ligado com sucesso.")

        ptu.voltar_origem()

        def seguir_face(bbox, frame_w, frame_h):
            if bbox is not None:
                ptu.track_face(bbox, frame_w, frame_h)
            else:
                ptu.reset_tracking()

        psainicio.seguir_face = seguir_face

        psainicio.run()

    except KeyboardInterrupt:
        print("Interrompido pelo utilizador.")

    except Exception as e:
        print("Erro no programa:", e)

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