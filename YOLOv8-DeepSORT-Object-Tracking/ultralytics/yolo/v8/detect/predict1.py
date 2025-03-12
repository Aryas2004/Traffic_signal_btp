import cv2
from ultralytics import YOLO

# Initialize counters
count1 = 0  # Count of cars near the signal (line 1)
count2 = 0  # Count of cars far from the signal (line 2)

# Define two horizontal lines for counting (y-coordinates)
LINE_1_Y = 300  # Line 1 near the signal
LINE_2_Y = 500  # Line 2 far from the signal

class Car:
    def __init__(self, car_id, bbox):
        self.car_id = car_id
        self.bbox = bbox  # (xmin, ymin, xmax, ymax)
        self.crossed_line_1 = False  # Initially, the car hasn't crossed line 1
        self.crossed_line_2 = False  # Initially, the car hasn't crossed line 2

# Load YOLOv8 model pre-trained on COCO (which includes 'car' class)
model = YOLO("yolov8n.pt")  # You can replace this with 'yolov8l.pt' or 'yolov8x.pt' for larger models

# Function to detect cars using YOLOv8
def detect_cars(frame):
    # Run the YOLO model on the frame
    results = model(frame)
    
    detected_cars = []
    car_id = 0

    # Extract boxes and labels from results
    for result in results[0].boxes:
        class_id = int(result.cls)  # Get the class ID
        if class_id == 2:  # Class 2 corresponds to 'car' in COCO dataset
            car_id += 1
            bbox = result.xyxy.numpy().astype(int)[0]  # (xmin, ymin, xmax, ymax)
            detected_cars.append(Car(car_id, bbox))
    
    return detected_cars

# Function to check if a car crosses line 1 (near the signal)
def car_crosses_line1(car):
    ymin, ymax = car.bbox[1], car.bbox[3]
    return ymin <= LINE_1_Y < ymax

# Function to check if a car crosses line 2 (far from the signal)
def car_crosses_line2(car):
    ymin, ymax = car.bbox[1], car.bbox[3]
    return ymin <= LINE_2_Y < ymax

# Main function to process video frames
def process_frame(frame):
    global count1, count2
    
    # Detect cars in the current frame
    detected_cars = detect_cars(frame)
    
    # Loop through all detected cars
    for car in detected_cars:
        # Check if the car crosses line 2 first (far from the signal)
        if not car.crossed_line_2 and car_crosses_line2(car):
            car.crossed_line_2 = True  # Mark that it crossed line 2
            count2 += 1  # Increment count at line 2

        # Check if the car that crossed line 2 also crosses line 1 (moving towards the signal)
        if car.crossed_line_2 and not car.crossed_line_1 and car_crosses_line1(car):
            car.crossed_line_1 = True  # Mark that it crossed line 1
            count1 += 1  # Increment count at line 1
    
    # Display counts and difference
    print(f"Cars counted at line 2 (far from signal): {count2}")
    print(f"Cars counted at line 1 (near the signal): {count1}")
    print(f"Cars moving towards the signal (line 2 -> line 1): {count2 - count1}")

# Main loop to process video
def main():
    video = cv2.VideoCapture('traffic_video.mp4')  # Replace with your video file
    
    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video
        
        # Process the frame to count cars
        process_frame(frame)
        
        # Draw the two lines on the frame for visualization
        cv2.line(frame, (0, LINE_1_Y), (frame.shape[1], LINE_1_Y), (0, 255, 0), 2)  # Line 1 (green)
        cv2.line(frame, (0, LINE_2_Y), (frame.shape[1], LINE_2_Y), (255, 0, 0), 2)  # Line 2 (blue)
        
        # Show the frame with lines
        cv2.imshow('Traffic Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()