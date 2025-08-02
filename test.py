import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

img_size = 224
batch_size = 32
test_dir = 'dataset/test'
model_path = 'model/fabric model_cnn.h5'

model = load_model(model_path)
print("✅ Model loaded successfully.")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_datagen.flow_from_directory(test_dir, target_size=(img_size, img_size),
                                             batch_size=batch_size, class_mode='categorical', shuffle=False)

class_labels = list(test_data.class_indices.keys())
true_classes = test_data.classes
filenames = test_data.filenames

pred_probs = model.predict(test_data)
pred_classes = np.argmax(pred_probs, axis=1)

for i in range(min(10, len(filenames))):
    pred_name = class_labels[pred_classes[i]]
    true_name = class_labels[true_classes[i]]
    confidence = np.max(pred_probs[i])
    print(f"Image: {filenames[i]} --> Predicted: {pred_name} (Conf: {confidence:.2f}) | True: {true_name}")

cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()

plt.show()

print("\n📊 Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=class_labels))