import cv2
import numpy
import lane_detection


def create_line(x, y):
    return numpy.array((x, y), dtype=numpy.int32)


def cv_line(frame, x, y):
    cv2.line(frame, tuple(x), tuple(y), (255, 255, 255), 2, lineType=cv2.LINE_AA)


video_capture = cv2.VideoCapture("./data/video/good_sample.mp4")

return_value, frame = video_capture.read()

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
line1_y = frame_height * 3 / 4
line2_y = frame_height / 2
line1 = create_line((0, line1_y), (frame_width, line1_y))
line2 = create_line((0, line2_y), (frame_width, line2_y))
line3_y = (line1_y + line2_y) / 2
line3 = create_line((0, line3_y), (frame_width, line3_y))

lane_det = lane_detection.LaneDetector(line1, line3, line2)
lane_det.draw(frame)

cv2.imshow("im", frame)
cv2.waitKey()
cv2.destroyAllWindows()
