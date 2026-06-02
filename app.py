import cv2
from ultralytics import YOLO


def main():
    # 1. Carrega o modelo YOLO (utilizando o YOLOv8 nano por ser leve e eficiente)
    # O arquivo .pt será baixado automaticamente na primeira execução
    model = YOLO("yolov8n.pt")

    # 2. Inicializa a captura de vídeo da câmera padrão (índice 0)
    # Se tiver uma câmera externa, pode ser necessário mudar para 1 ou 2
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Scanner YOLO iniciado. Pressione 'q' para sair.")

    # 3. Loop principal para processamento dos frames em tempo real
    while True:
        # Captura o frame atual da câmera
        ret, frame = cap.read()

        if not ret:
            print("Erro: Falha ao receber o frame da câmera.")
            break

        # 4. Executa a inferência do YOLO no frame capturado
        # stream=True otimiza o uso de memória para processamento de vídeo
        results = model(frame, stream=True)

        # 5. Renderiza as caixas delimitadoras e labels no frame atual
        for result in results:
            annotated_frame = result.plot()

            # Exibe o frame processado em uma janela nativa do sistema operacional
            cv2.imshow("Scanner com YOLO (Pressione 'q' para sair)", annotated_frame)

        # 6. Condição de parada: interrompe o loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 7. Libera os recursos de hardware e fecha as janelas criadas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()