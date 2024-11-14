import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("Tips for Better Results:")
print("- Use a plain background.")
print("- Ensure a single, strong light source to create clear shadows.")
print("- Place your object or hand close to the light for better detection.")

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast and brightness
    enhanced_frame = cv2.convertScaleAbs(gray_frame, alpha=1.5, beta=30)

    # Blur to reduce noise
    blurred_frame = cv2.GaussianBlur(enhanced_frame, (7, 7), 0)

    # Use adaptive thresholding for more precise mask
    binary_mask = cv2.adaptiveThreshold(
        blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Create a blank white canvas for shadow dots
    dot_canvas = np.ones_like(clean_mask) * 255

    # Draw dots for detected shadow areas
    dot_size = 5  # Size of each dot
    for y in range(0, clean_mask.shape[0], dot_size):  # Dense grid of dots
        for x in range(0, clean_mask.shape[1], dot_size):
            if clean_mask[y, x] == 255:  # If shadow detected
                cv2.circle(dot_canvas, (x, y), dot_size // 2, (0, 0, 0), -1)

    # Display results
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Processed Shadow", clean_mask)
    cv2.imshow("Shadow Dots", dot_canvas)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()













