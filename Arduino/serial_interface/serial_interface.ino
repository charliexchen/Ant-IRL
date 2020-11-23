#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
int incomingByte = 0;    // for incoming serial data
int currentCommands[8] = {750, 750, 750, 750, 750, 750, 750, 750}; // initial servo positions
int servoPositionOffset = 8;

unsigned int raw_command = 0;
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

int granularity = 2;

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
        int servo_pos = raw_command >> 3;
        currentCommands[servo_id] = servo_pos;
        raw_command = 0;
      }
    }
  }
  for (int i = 0; i < 8; i++) {
   uint16_t raw_servo_id = i + servoPositionOffset;
    uint16_t raw_ms_command = currentCommands[i]*granularity;
Serial.println(String(raw_servo_id) + ", " +String(raw_ms_command));
    pwm.writeMicroseconds(raw_servo_id, raw_ms_command);
  }
  delay(50);
}
