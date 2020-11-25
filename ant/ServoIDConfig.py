from enum import Enum
from typing import Union

MAX_PULSE = 2400
MIN_PULSE = 600
MAX_SERVO_COUNT = 8


class Servos(Enum):
    FRONT_RIGHT_HIP = "front_right_hip"
    FRONT_LEFT_HIP = "front_left_hip"
    BACK_LEFT_HIP = "back_left_hip"
    BACK_RIGHT_HIP = "back_right_hip"
    FRONT_RIGHT_LEG = "front_right_leg"
    FRONT_LEFT_LEG = "front_left_leg"
    BACK_LEFT_LEG = "back_left_leg"
    BACK_RIGHT_LEG = "back_right_leg"


INVERTED_SERVOS = {
    Servos.FRONT_RIGHT_HIP,
    Servos.BACK_RIGHT_HIP,
    Servos.FRONT_RIGHT_LEG,
    Servos.BACK_LEFT_LEG
}

SERVO_ENUM_TO_ID_MAP = {"front_right_hip": 0,
                        "front_left_hip": 1,
                        "back_left_hip": 2,
                        "back_right_hip": 3,
                        "front_right_leg": 4,
                        "front_left_leg": 5,
                        "back_left_leg": 6,
                        "back_right_leg": 7, }


def get_servo_id(servo: Union[Servos, str, int]) -> int:
    if type(servo) is int:
        return servo
    if servo in SERVO_ENUM_TO_ID_MAP:
        return SERVO_ENUM_TO_ID_MAP[servo]
    try:
        return SERVO_ENUM_TO_ID_MAP[servo.value]
    except AttributeError:
        return False


INVERTED_SERVO_IDS = {get_servo_id(servo) for servo in INVERTED_SERVOS}


def is_inverted(servo: Union[int, Servos]) -> bool:
    return servo in INVERTED_SERVO_IDS or servo in INVERTED_SERVOS
