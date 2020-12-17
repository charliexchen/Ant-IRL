typedef struct {
 
 float ax1;
 float ay1;
  char al;

} __attribute__((__packed__))data_packet_t;

data_packet_t dp;

template <typename T> void sendData(T data)
{
 Serial.write((byte*)&data, sizeof(data));
}
void setup() {
  Serial.begin(57600);
}
void loop(){
  dp.al = "a";
dp.ax1 = 0.03; 
dp.ay1 = 0.3;
sendData<data_packet_t>(dp);
delay(5);
}
