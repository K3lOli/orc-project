from flask import Flask, request, jsonify
import cv2
import pytesseract
from PIL import Image
import numpy as np
import io
import nltk
from nltk.tokenize import sent_tokenize
from flask_cors import CORS

nltk.download('punkt')

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

app = Flask(__name__)

CORS(app)

@app.route('/extract_text', methods=['POST'])
def extract_text():
    try:
        # Obtenha a imagem do formulário HTML
        imagem = request.files['image'].read()

        # Converta a imagem em um objeto de imagem PIL
        img = Image.open(io.BytesIO(imagem))

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converta para tons de cinza
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Binarize a imagem

        # Extraia o texto da imagem
        text = pytesseract.image_to_string(img , lang='por')

        # texto_extraido = texto_extraido.strip()

        # Exiba o texto no terminal do servidor
        # Use a função sent_tokenize para dividir o texto em frases
        sentences = sent_tokenize(text)
        
        # Combine as frases em parágrafos com quebras de linha entre elas
        formatted_text = '\n\n'.join(sentences)

        print("Texto Extraído:")
        print(formatted_text)

        # Retorne o texto como resposta para o cliente
        return jsonify({'text': formatted_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
