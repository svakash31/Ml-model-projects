import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
classes = ["No Helmet", "Helmet"]  
COLORS = {"Helmet": (0, 255, 0), "No Helmet": (0, 0, 255)}

# Load video or webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Preprocessing image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Process detection outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold if needed
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                
                if class_id < len(classes):
                    class_ids.append(class_id)
                else:
                    class_ids.append(0)  # Default to "No Helmet" if out of bounds

    # Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    ignition_status = "Ignition Off"  # Default to Off
    largest_area = 0
    selected_box, selected_label = None, None

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            area = w * h  # Calculate bounding box area
            
            # Select only the largest detected person
            if area > largest_area:
                largest_area = area
                selected_box = (x, y, w, h)
                selected_label = classes[class_ids[i]]

    # Draw bounding box and update ignition status only for the largest person
    if selected_box:
        x, y, w, h = selected_box
        color = COLORS[selected_label]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{selected_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if selected_label == "Helmet":
            ignition_status = "Ignition On"

    # Display ignition status
    cv2.putText(frame, ignition_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Display result
    cv2.imshow("Helmet Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
