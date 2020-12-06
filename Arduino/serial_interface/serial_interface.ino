#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>


Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
int incomingByte = 0;    // for incoming serial data
int currentCommands[8] = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500}; // initial servo positions
int servoPositionOffset = 8;

int maxPulse = 2400;
int minPulse = 600;

unsigned int raw_command = 0;
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

int granularity = 1;

void setup() {
  Serial.begin(9600);    // opens serial port, sets data rate to 9600 bps

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

  delay(5);
}

void loop() {
  if (Serial.available() > 0) {
    while (Serial.available() > 0) {
      incomingByte = Serial.read();
      raw_command = (raw_command << 7) | (incomingByte);
      bool is_last_bit = (incomingByte & 128) == 128;
      if (is_last_bit) {
        int servo_id = raw_command & 7;
        int servo_pos = (raw_command >> 3) + minPulse;
        if (servo_pos > maxPulse) {
          servo_pos = maxPulse;
        }
        currentCommands[servo_id] = servo_pos;
        raw_command = 0;
      }
    }
  }
  for (int i = 0; i < 8; i++) {
    uint16_t raw_servo_id = i + servoPositionOffset;
    uint16_t raw_ms_command = (currentCommands[i] * granularity);
    pwm.writeMicroseconds(raw_servo_id, raw_ms_command);
  }
  delay(10);
  pwm.writeMicroseconds(0, 1500);
}
