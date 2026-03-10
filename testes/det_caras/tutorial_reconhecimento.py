import cv2  # Importa a biblioteca OpenCV, usada para visão por computador

# Função que calcula uma pontuação de qualidade/confiança entre 0 e 1
def calcular_confianca(face_roi_gray):
    # Calcula a nitidez da imagem usando a variância do Laplaciano
    nitidez = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()

    # Calcula o brilho médio da região da cara
    brilho = face_roi_gray.mean()

    # Obtém a altura e a largura da região da cara
    altura, largura = face_roi_gray.shape

    # Calcula a área da cara
    area = largura * altura

    # Normaliza a nitidez para um valor entre 0 e 1
    score_nitidez = min(nitidez / 500, 1.0)

    # Calcula uma pontuação de brilho, favorecendo valores próximos de 128
    score_brilho = 1.0 - abs(brilho - 128) / 128

    # Garante que o valor do brilho fica entre 0 e 1
    score_brilho = max(0.0, min(score_brilho, 1.0))

    # Normaliza a área da cara para um valor entre 0 e 1
    score_area = min(area / 20000, 1.0)

    # Faz uma média ponderada dos três critérios
    confianca = 0.5 * score_nitidez + 0.2 * score_brilho + 0.3 * score_area

    # Garante que a confiança final fica entre 0 e 1
    confianca = max(0.0, min(confianca, 1.0))

    # Devolve o valor final da confiança
    return confianca


# Cria um classificador de rostos com um ficheiro Haar Cascade já incluído no OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # Caminho para o classificador de caras frontais
)

# Abre a webcam principal do computador
cap = cv2.VideoCapture(0)

# Verifica se a câmara abriu corretamente
if not cap.isOpened():
    print("Erro: não foi possível abrir a câmara.")  # Mostra mensagem de erro
    exit()  # Termina o programa

# Mostra mensagem informativa no terminal
print("Câmara aberta. Carrega em 'q' para sair.")

# Ciclo infinito para capturar continuamente frames da webcam
while True:
    ret, frame = cap.read()  # Lê um frame da câmara

    # Se não conseguiu ler o frame, termina o ciclo
    if not ret:
        print("Erro ao ler imagem da câmara.")
        break

    # Converte a imagem a cores para tons de cinzento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteta caras na imagem em tons de cinzento
    faces = face_cascade.detectMultiScale(
        gray,  # Imagem em cinzento
        scaleFactor=1.1,  # Procura caras em diferentes escalas
        minNeighbors=5,  # Número mínimo de vizinhos para validar uma deteção
        minSize=(60, 60)  # Tamanho mínimo de cara a detetar
    )

    # Inicializa a maior área encontrada
    maior_area = 0

    # Inicializa a variável que vai guardar a maior cara
    maior_face = None

    # Percorre todas as caras detetadas
    for (x, y, w, h) in faces:
        area = w * h  # Calcula a área da cara atual

        # Se esta cara for maior do que a maior encontrada até agora
        if area > maior_area:
            maior_area = area  # Atualiza a maior área
            maior_face = (x, y, w, h)  # Guarda esta cara como a maior

    # Percorre novamente todas as caras para desenhar as caixas
    for (x, y, w, h) in faces:
        margem = 20  # Define uma margem extra à volta da caixa

        # Calcula o canto superior esquerdo da caixa alargada
        x1 = max(x - margem, 0)

        # Calcula o y superior da caixa alargada
        y1 = max(y - margem, 0)

        # Calcula o canto inferior direito da caixa alargada
        x2 = min(x + w + margem, frame.shape[1])

        # Calcula o y inferior da caixa alargada
        y2 = min(y + h + margem, frame.shape[0])

        # Se esta cara for a maior, usa azul
        if maior_face == (x, y, w, h):
            cor = (255, 0, 0)  # Azul em formato BGR
        else:
            cor = (0, 255, 0)  # Verde em formato BGR

        # Recorta a região da cara na imagem em tons de cinzento
        face_roi_gray = gray[y:y+h, x:x+w]

        # Verifica se a região recortada existe
        if face_roi_gray.size > 0:
            confianca = calcular_confianca(face_roi_gray)  # Calcula a confiança/qualidade
        else:
            confianca = 0.0  # Se a região estiver vazia, assume confiança 0

        # Desenha o retângulo à volta da cara
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        # Cria o texto da confiança com 2 casas decimais
        texto = f"({confianca:.2f})"

        # Calcula o tamanho do texto para o posicionar corretamente
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Calcula a posição x do texto no canto inferior direito da caixa
        text_x = x2 - tw - 5

        # Calcula a posição y do texto um pouco acima do limite inferior da caixa
        text_y = y2 - 5

        # Escreve o texto da confiança na imagem
        cv2.putText(
            frame,  # Imagem onde o texto vai ser escrito
            texto,  # Texto a mostrar
            (text_x, text_y),  # Posição do texto
            cv2.FONT_HERSHEY_SIMPLEX,  # Tipo de letra
            0.6,  # Escala do texto
            cor,  # Cor do texto
            2  # Espessura da letra
        )

    # Mostra a imagem com as caixas e os valores de confiança
    cv2.imshow("Deteção de Caras", frame)

    # Espera 1 ms por uma tecla e termina se a tecla for 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberta a webcam
cap.release()

# Fecha todas as janelas abertas pelo OpenCV
cv2.destroyAllWindows()