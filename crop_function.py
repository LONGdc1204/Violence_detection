import math
import cv2
from model_yolov8 import model_yolo
from save_frames_to_video import convert_video_to_small_videos


def crop(path_in, path_out):
    count = 0
    cap = cv2.VideoCapture(path_in)
    frame_array = []
    while True:
        success, img = cap.read()

        if success:
            count += 1
            results = model_yolo(img, stream=True)
            for r in results:
                boxes = r.boxes
                person_boxes = []
                conf_boxes = []
                for box in boxes:
                    if int(box.cls[0]) == 0:
                        person_boxes.append(box.xyxy[0])
                        conf_boxes.append(box.conf[0])

                if len(person_boxes) >= 2:
                    x1, y1, x2, y2 = person_boxes[0]
                    X1, Y1, X2, Y2 = person_boxes[1]
                    conf_person1, conf_person2 = math.ceil((conf_boxes[0] * 100)) / 100, math.ceil(
                        (conf_boxes[1] * 100)) / 100

                if 1 <= len(person_boxes) < 2:
                    x1, y1, x2, y2 = person_boxes[0]
                    X1, Y1, X2, Y2 = x1, y1, x2, y2
                    conf_person1 = math.ceil((conf_boxes[0] * 100)) / 100
                    conf_person2 = conf_person1

                if len(person_boxes) == 0:
                    x1, y1, x2, y2 = 0, 0, 0, 0
                    X1, Y1, X2, Y2 = x1, y1, x2, y2
                    conf_person1 = 0
                    conf_person2 = conf_person1

                pos_x, pos_y, pos_X, pos_Y = min(x1, X1), min(y1, Y1), max(x2, X2), max(y2, Y2)
                pos_x, pos_y, pos_X, pos_Y = (int(pos_x - 1 / 10 * pos_X), int(pos_y - 1 / 10 * pos_y),
                                              int(pos_X + 1 / 10 * pos_X), int(pos_Y + 1 / 10 * pos_Y))

                if conf_person1 >= 0.6 and conf_person2 >= 0.6:
                    crop_img = img[pos_y:pos_Y, pos_x:pos_X]
                    if crop_img.size > 0:
                        crop_img = cv2.resize(crop_img, (256, 256))
                        frame_array.append(crop_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()

    print(len(frame_array))
    print(count)
    frames = []
    for i in range(len(frame_array)):
        frames.append(frame_array[i])
        if len(frames) == 32:
            convert_video_to_small_videos(frames, frame_size=(256, 256), path_out=path_out,
                                          count=(i + 1) // 32)
            frames.clear()
