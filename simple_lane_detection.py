from lane_detection import MovementDirection
from enum import Enum
import cv2


class TrackDirection(Enum):
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'


class SimpleLaneCounter:
    def __init__(self, delimiter, is_draw=True,
                 movement_direction=MovementDirection.VERTICAL):
        self.delimiter = delimiter
        self.movement_direction = movement_direction
        self.is_draw = is_draw
        self.up_crossed = 0
        self.down_crossed = 0
        self.right_crossed = 0
        self.left_crossed = 0

    def get_y(self):
        return self.delimiter[0][1]

    def get_x(self):
        return self.delimiter[0][0]

    def process_track(self, track):
        x, y = track.mean[:2]
        if MovementDirection.VERTICAL == self.movement_direction:
            if track.direction == TrackDirection.UP and y < self.get_y() < track.init_pos[1]:
                track.counted = True
                self.up_crossed += 1
            elif track.direction == TrackDirection.DOWN and y > self.get_y() > track.init_pos[1]:
                track.counted = True
                self.down_crossed += 1
        else:
            if track.direction == TrackDirection.RIGHT and x > self.get_x():
                track.counted = True
                self.right_crossed += 1
            elif track.direction == TrackDirection.LEFT and x < self.get_x():
                track.counted = True
                self.left_crossed += 1

    def draw_line(self, frame):
        cv2.line(frame, tuple(self.delimiter[0]), tuple(self.delimiter[1]), (255, 255, 255), 2, lineType=cv2.LINE_AA)

    def draw_statistic(self, frame, frame_width, fps, frame_number):
        if MovementDirection.VERTICAL == self.movement_direction:
            self.draw_statistic_vertical(frame, frame_width, fps, frame_number)

    def draw_statistic_vertical(self, frame, frame_width, fps, frame_number):
        time_passed_s = frame_number / fps
        time_passed_m = time_passed_s / 60
        count_up_in_min = int(self.up_crossed / time_passed_m)
        count_down_in_min = int(self.down_crossed / time_passed_m)
        color = (46, 0, 115)
        cv2.putText(frame, "Left line: " + str(self.up_crossed), (150, 100),
                    fontScale=1.5, fontFace=cv2.FONT_ITALIC,
                    color=color, thickness=2
                    )
        cv2.putText(frame, "Right line: " + str(self.down_crossed), (frame_width - 400, 100),
                    fontScale=1.5, fontFace=cv2.FONT_ITALIC,
                    color=color, thickness=2
                    )
        cv2.putText(frame, "LL Density: " + str(count_up_in_min), (150, 200),
                    fontScale=1.5, fontFace=cv2.FONT_ITALIC,
                    color=color, thickness=2
                    )
        cv2.putText(frame, "RL Density: " + str(count_down_in_min), (frame_width - 400, 200),
                    fontScale=1.5, fontFace=cv2.FONT_ITALIC,
                    color=color, thickness=2
                    )
