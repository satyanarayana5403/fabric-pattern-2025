from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create upload directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = 'model/fabric model_cnn.h5'
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Class labels
class_labels = ['floral fabric pattern', 'geometric fabric', 'polka dot fabric', 'striped fabric']

# Augmentation generator (for preview image)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Augment one preview image
            aug_iter = datagen.flow(img_array, batch_size=1)
            aug_img = next(aug_iter)[0]
            aug_filename = f"aug_{filename}"
            aug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], aug_filename)
            # Convert [-1,1] to [0,255]
            aug_img_uint8 = np.clip(((aug_img + 1) * 127.5), 0, 255).astype(np.uint8)
            Image.fromarray(aug_img_uint8).save(aug_filepath)

            # Predict
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            predicted_class = class_labels[pred_index]
            confidence = predictions[0][pred_index]
            confidence_dict = {label: float(f"{conf:.4f}") for label, conf in zip(class_labels, predictions[0])}

            return render_template('predict.html',
                                   filename=filename,
                                   aug_filename=aug_filename,
                                   prediction=predicted_class,
                                   confidence=confidence,
                                   scores=confidence_dict)

    return render_template('predict.html', filename=None, aug_filename=None, prediction=None, confidence=None, scores=None)

if __name__ == '__main__':
    app.run(debug=True)
