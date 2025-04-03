import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TensorFlow Lite model
model_path = "model.tflite"  # Ensure your .tflite model is in the same directory
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
input_shape = input_details[0]['shape']
img_size = (input_shape[1], input_shape[2])  # Expected input size (300, 300)

# Define class labels (Ensure these match your model's classes)
labels = {0: 'ALL_MOTOR_VEHICLE_PROHIBITED', 1: 'AXLE_LOAD_LIMIT', 2: 'BARRIER_AHEAD', 3: 'BULLOCK_AND_HANDCART_PROHIBITED', 
          4: 'CATTLE', 5: 'COMPULSARY_AHEAD', 6: 'COMPULSARY_AHEAD_OR_TURN_LEFT', 7: 'COMPULSARY_AHEAD_OR_TURN_RIGHT',
          8: 'COMPULSARY_CYCLE_TRACK', 9: 'COMPULSARY_KEEP_RIGHT', 10: 'COMPULSARY_MINIMUM_SPEED', 11: 'COMPULSARY_SOUND_HORN',
          12: 'COMPULSARY_TURN_LEFT', 13: 'COMPULSARY_TURN_LEFT_AHEAD', 14: 'COMPULSARY_TURN_RIGHT', 15: 'COMPULSARY_TURN_RIGHT_AHEAD',
          16: 'CROSS_ROAD', 17: 'CYCLE_CROSSING', 18: 'CYCLE_PROHIBITED', 19: 'DANGEROUS_DIP', 20: 'DIRECTION', 21: 'FALLING_ROCKS',
          22: 'FERRY', 23: 'GAP_IN_MEDIAN', 24: 'GIVE_WAY', 25: 'GUARDED_LEVEL_CROSSING', 26: 'HANDCART_PROHIBITED', 
          27: 'HEIGHT_LIMIT', 28: 'HORN_PROHIBITED', 29: 'HUMP_OR_ROUGH_ROAD', 30: 'LEFT_HAIR_PIN_BEND', 31: 'LEFT_HAND_CURVE', 
          32: 'LEFT_REVERSE_BEND', 33: 'LEFT_TURN_PROHIBITED', 34: 'LENGTH_LIMIT', 35: 'LOAD_LIMIT', 36: 'LOOSE_GRAVEL', 
          37: 'MEN_AT_WORK', 38: 'NARROW_BRIDGE', 39: 'NARROW_ROAD_AHEAD', 40: 'NO_ENTRY', 41: 'NO_PARKING', 42: 'NO_STOPPING_OR_STANDING',
          43: 'OVERTAKING_PROHIBITED', 44: 'PASS_EITHER_SIDE', 45: 'PEDESTRIAN_CROSSING', 46: 'PEDESTRIAN_PROHIBITED', 47: 'PRIORITY_FOR_ONCOMING_VEHICLES',
          48: 'QUAY_SIDE_OR_RIVER_BANK', 49: 'RESTRICTION_ENDS', 50: 'RIGHT_HAIR_PIN_BEND', 51: 'RIGHT_HAND_CURVE', 52: 'RIGHT_REVERSE_BEND',
          53: 'RIGHT_TURN_PROHIBITED', 54: 'ROAD_WIDENS_AHEAD', 55: 'ROUNDABOUT', 56: 'SCHOOL_AHEAD', 57: 'SIDE_ROAD_LEFT',
          58: 'SIDE_ROAD_RIGHT', 59: 'SLIPPERY_ROAD', 60: 'SPEED_LIMIT_15', 61: 'SPEED_LIMIT_20', 62: 'SPEED_LIMIT_30',
          63: 'SPEED_LIMIT_40', 64: 'SPEED_LIMIT_5', 65: 'SPEED_LIMIT_50', 66: 'SPEED_LIMIT_60', 67: 'SPEED_LIMIT_70',
          68: 'SPEED_LIMIT_80', 69: 'STAGGERED_INTERSECTION', 70: 'STEEP_ASCENT', 71: 'STEEP_DESCENT', 72: 'STOP', 73: 'STRAIGHT_PROHIBITED',
          74: 'TONGA_PROHIBITED', 75: 'TRAFFIC_SIGNAL', 76: 'TRUCK_PROHIBITED', 77: 'TURN_RIGHT', 78: 'T_INTERSECTION', 
          79: 'UNGUARDED_LEVEL_CROSSING', 80: 'U_TURN_PROHIBITED', 81: 'WIDTH_LIMIT', 82: 'Y_INTERSECTION'}



# Initialize the Raspberry Pi Camera
cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Preprocess the image
    img = cv2.resize(frame, img_size)  # Resize to model input size (300x300)
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class and confidence
    predicted_class = np.argmax(output_data[0])
    confidence = np.max(output_data[0])

    # Display results on the frame
    label_text = f"{labels.get(predicted_class, 'Unknown')} ({confidence:.2f})"
    cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Traffic Sign Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
