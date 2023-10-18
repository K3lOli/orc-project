import cv2

# Inicialize a captura de vídeo a partir da webcam
cap = cv2.VideoCapture(0)

# Carregue o modelo de classificação pré-treinado para detecção de rosto
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

while True:
    # Captura um quadro da webcam
    ret, frame = cap.read()
    
    # Verifique se a captura de vídeo foi bem-sucedida
    if not ret:
        break
    
    # Converta o quadro para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecte rostos no quadro
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Desenhe um retângulo em torno do rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Verifique se uma mão é levantada (você pode ajustar os valores)
    if len(faces) > 0 and y < 200:
        cv2.putText(frame, "Mao Levantada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Mostre o quadro
    cv2.imshow("Detecao de Mao", frame)
    
    # Saia do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura de vídeo e feche a janela
cap.release()
cv2.destroyAllWindows()
