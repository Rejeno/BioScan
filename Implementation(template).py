import os
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from pyzkfp import ZKFP2
import fingerprint_enhancer  # Fingerprint enhancement library
import time  # ⏳ Add delay before re-scanning

# Load the trained model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH, compile=False)

# Blood group classes
classes = ['A', 'AB','B','O']


# Ensure upload directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def check_image_quality(image):
    """
    Check if the fingerprint image has sufficient quality for enhancement.
    """
    if image.size == 0:
        print("❌ Error: Image is empty.")
        return False

    # Check if the image is too dark or too bright
    mean_intensity = np.mean(image)
    if mean_intensity < 30 or mean_intensity > 220:
        print(f"❌ Error: Image intensity is out of range (mean={mean_intensity:.2f}).")
        return False

    # Check if the image has sufficient contrast
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    if max_intensity - min_intensity < 50:
        print(f"❌ Error: Image contrast is too low (range={max_intensity - min_intensity}).")
        return False

    return True


def enhance_fingerprint(image):
    """
    Enhance the fingerprint image using the `fingerprint_enhancer` library.
    """
    # Convert to grayscale (if not already)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check image quality
    if not check_image_quality(image):
        return None

    # Enhance the fingerprint using the `fingerprint_enhancer` library
    try:
        enhanced_image = fingerprint_enhancer.enhance_fingerprint(image)
    except Exception as e:
        print(f"❌ Error during fingerprint enhancement: {e}")
        return None

    # Convert binary image to grayscale (0-255)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image


def preprocess_fingerprint(image_array):
    """
    Preprocess the fingerprint image before passing it to the model.
    - Convert to grayscale if needed
    - Resize to 128x128 (model input size)
    - Normalize pixel values (0-1)
    - Ensure shape is (1, 128, 128, 1) for model compatibility
    """
    # Enhance the fingerprint image
    enhanced_image = enhance_fingerprint(image_array)
    if enhanced_image is None:
        return None

    # Resize to model input size
    image = cv2.resize(enhanced_image, (300, 400))

    # Normalize pixel values
    image = img_to_array(image) / 255.0

    # Add batch dimension and ensure shape (1, 128, 128, 1)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    return image


def classify_fingerprint(image_array):
    """
    Runs fingerprint image through the trained model and returns classification result.
    """
    image = preprocess_fingerprint(image_array)
    if image is None:
        return None, None

    predictions = model.predict(image)
    predicted_index = np.argmax(predictions)
    predicted_class = classes[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    return predicted_class, confidence


def main():
    """
    Main loop that continuously waits for a fingerprint scan,
    processes it, classifies it, and displays the result.
    """
    print("\n🔄 Initializing fingerprint scanner...")

    # Initialize fingerprint scanner
    zkfp2 = ZKFP2()
    if zkfp2.Init():
        print("❌ Failed to initialize fingerprint scanner. Exiting.")
        return

    device_count = zkfp2.GetDeviceCount()
    if device_count == 0:
        print("❌ No fingerprint scanner found! Connect the device.")
        zkfp2.Terminate()
        return

    print(f"✅ {device_count} fingerprint scanner(s) detected.")
    zkfp2.OpenDevice(0)

    print("\n👉 Please place your finger on the scanner...")  # Show only once

    try:
        while True:
            # Wait for a fingerprint scan
            capture = zkfp2.AcquireFingerprint()
            if not capture:
                continue  # If no fingerprint is detected, keep waiting

            tmp, img = capture
            if not img or len(img) == 0:
                print("❌ Error: Fingerprint image is empty. Try again.")
                continue

            # Convert raw fingerprint data to image format
            img_array = np.frombuffer(img, dtype=np.uint8)
            actual_size = len(img_array)

            # Detect fingerprint image size dynamically
            if actual_size == 112500:  # 300x375
                detected_height, detected_width = 375, 300
            elif actual_size == 120000:  # 300x400
                detected_height, detected_width = 400, 300
            else:
                print(f"❌ Unknown image size ({actual_size} bytes). Skipping...")
                continue

            fingerprint_image = img_array.reshape((detected_height, detected_width))

            # Save fingerprint for debugging (optional)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(UPLOAD_FOLDER, f"fingerprint_{timestamp}.bmp")
            cv2.imwrite(file_path, fingerprint_image)

            print("\n✅ Fingerprint captured! Processing...")

            # Check image quality
            if not check_image_quality(fingerprint_image):
                print("❌ Poor image quality. Please place your finger on the scanner again.")
                continue  # Skip to the next iteration

            # Enhance the fingerprint image
            enhanced_image = enhance_fingerprint(fingerprint_image)
            if enhanced_image is None:
                print("❌ Fingerprint enhancement failed. Please place your finger on the scanner again.")
                continue  # Skip to the next iteration

            # Save enhanced fingerprint for debugging (optional)
            enhanced_file_path = os.path.join(UPLOAD_FOLDER, f"enhanced_fingerprint_{timestamp}.bmp")
            cv2.imwrite(enhanced_file_path, enhanced_image)

            # Classify fingerprint
            predicted_class, confidence = classify_fingerprint(enhanced_image)
            if predicted_class is None:
                print("❌ Fingerprint classification failed. Please place your finger on the scanner again.")
                continue  # Skip to the next iteration

            # Display result
            print("\n🔬 **Classification Result** 🔬")
            print(f"🩸 Predicted Blood Group: {predicted_class}")
            print(f"📊 Confidence: {confidence:.2f}%")

            print("\n🔄 Waiting for next fingerprint...")  # Instead of re-showing instructions

            # Wait before scanning next fingerprint (Prevents double detection)
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n🔌 Exiting program...")

    finally:
        zkfp2.Terminate()


if __name__ == "__main__":
    main()