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

typedef struct {
  float qw = 0;
  float qx = 0;
  float qy = 0;
  float qz = 0;
  float y = 0;
  float p = 0;
  float r = 0;
  float gx = 0;
  float gy = 0;
  float gz = 0;
  float ed = 99999999.9;
} __attribute__((__packed__))data_packet_t;

data_packet_t dp;


void setup() {
  Serial.begin(9600);    // opens serial port, sets data rate to 9600 bps

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

  delay(300);
  Serial.println("Ant Ready!");
  delay(300);
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
  Serial.write((byte*)&dp, sizeof(dp));
  delay(10);
}
