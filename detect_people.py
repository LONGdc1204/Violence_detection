import math

from ultralytics import YOLO
import cv2


model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('Videos/test_datn_3.avi')
frame_array = []
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1080, 720))
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        person_boxes = []
        for index, box in enumerate(boxes):
            if int(box.cls[0]) == 0:
                person_boxes.append(box.xyxy[0])

        for box in boxes:
            if len(person_boxes) >=2:
                x1, y1, x2, y2 = person_boxes[0]
                X1, Y1, X2, Y2 = person_boxes[1]
                pos_x, pos_y, pos_X, pos_Y = min(x1, X1), min(y1, Y1), max(x2, X2), max(y2, Y2)
                pos_x, pos_y, pos_X, pos_Y = int(pos_x - 1/10 * pos_X), int(pos_y - 1/10 * pos_y), int(pos_X + 1/10 * pos_X), int(pos_Y + 1/10 * pos_Y)

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])

                if cls == 0 and conf >= 0.6:
                    cv2.rectangle(img, (pos_x, pos_y), (pos_X, pos_Y), (255, 0, 255), 2)
                    crop = img[pos_y:pos_Y, pos_x:pos_X]
                    crop = cv2.resize(crop, (256, 256))
                    frame_array.append(crop)
    cv2.imshow("Image", img)
    cv2.waitKey(1)