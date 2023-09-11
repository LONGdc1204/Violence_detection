import cv2


def convert_video_to_small_videos(frame_array, frame_size, path_out,  count):
    out = cv2.VideoWriter(path_out + f"_{count}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 60, frame_size)
    for i in range(0, len(frame_array)):
        out.write(frame_array[i])

    print(f"done {path_out}_{count}.avi")
    out.release()