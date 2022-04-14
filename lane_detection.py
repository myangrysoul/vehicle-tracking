from enum import Enum

import cv2


class MovementDirection(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class LaneType(Enum):
    HIGHER_COORD_VALUE = 1
    MIDDLE = 2
    LOWER_COORD_VALUE = -1


class Lane:
    def __init__(self, linepts, type, movement_direction):
        if movement_direction == MovementDirection.VERTICAL:
            self.const_value = linepts[0][1]
        else:
            self.const_value = linepts[0][0]
        self.type = type


class LaneDetector:
    def __init__(self, higher_coord_m_line, middle_m_line, lower_coord_m_line, is_draw=True,
                 movement_direction=MovementDirection.VERTICAL):
        self.bottom_m_lane = Lane(higher_coord_m_line, LaneType.HIGHER_COORD_VALUE, movement_direction)
        self.middle_m_lane = Lane(middle_m_line, LaneType.MIDDLE, movement_direction)
        self.top_m_lane = Lane(lower_coord_m_line, LaneType.LOWER_COORD_VALUE, movement_direction)
        self.lines = [higher_coord_m_line, middle_m_line, lower_coord_m_line]
        self.object_lanes = [self.bottom_m_lane, self.middle_m_lane, self.top_m_lane]
        self.movement_direction = movement_direction
        self.is_draw = is_draw

    def get_oposite(self, linetype):
        if linetype == LaneType.HIGHER_COORD_VALUE:
            return self.top_m_lane
        elif linetype == LaneType.LOWER_COORD_VALUE:
            return self.bottom_m_lane
        else:
            return self.top_m_lane

    def is_on_oposite_side_now(self, linetype, x, y):
        line = self.get_oposite(linetype)
        if self.movement_direction == MovementDirection.VERTICAL:
            if line.type == LaneType.HIGHER_COORD_VALUE:
                return line.const_value > y > self.middle_m_lane.const_value
            else:
                return line.const_value < y < self.middle_m_lane.const_value
        else:
            return line.const_value > x > self.middle_m_lane.const_value

    def in_bounds(self, x, y):
        if self.movement_direction == MovementDirection.VERTICAL:
            return self.bottom_m_lane.const_value > y > self.top_m_lane.const_value
        else:
            return self.bottom_m_lane.const_value > x > self.top_m_lane.const_value

    def get_closest(self, x, y):
        closest = None
        min_distance = 10000
        for i, lane in enumerate(self.object_lanes):
            if lane.type == LaneType.MIDDLE:
                continue
            distance = abs(lane.const_value - y)
            if distance < min_distance:
                min_distance = distance
                closest = lane
        return closest.type

    def process_track(self, track):
        x, y = track.mean[:2]
        if track.crossed_line is not None:
            if self.is_on_oposite_side_now(track.crossed_line, x, y):
                to_return = track.crossed_line.value
                track.crossed_line = None
                track.counted = True
                return to_return
        else:
            if self.in_bounds(x, y):
                track.crossed_line = self.get_closest(x, y)
                return 0

    def draw(self, frame):
        for i, line in enumerate(self.lines):
            cv2.line(frame, tuple(line[0]), tuple(line[1]), (255, 255, 255), 2, lineType=cv2.LINE_AA)
