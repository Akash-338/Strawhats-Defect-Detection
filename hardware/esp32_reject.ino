#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>

#define SERVO_PIN       27
#define BUZZER_PIN      32
#define RELAY_CH1_PIN   33
#define RELAY_CH2_PIN   25
#define LED_PASS_PIN    2
#define LED_REJECT_PIN  4

#define SERVO_HOME      0
#define SERVO_REJECT    90
#define SERVO_SPEED_MS  15

#define RELAY_ACTIVE_LOW true

#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT   64
#define OLED_RESET      -1
#define OLED_ADDR       0x3C

#define REJECT_HOLD_MS  3000   
#define TEST_DELAY_MS   3000

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
Servo rejectServo;

String currentMaterial = "---";
int    currentDefectCount = 0;
bool   isRejected = false;
unsigned long rejectStartTime = 0;
int    servoPos = SERVO_HOME;

void relayOn(int pin) {
  digitalWrite(pin, RELAY_ACTIVE_LOW ? LOW : HIGH);
}

void relayOff(int pin) {
  digitalWrite(pin, RELAY_ACTIVE_LOW ? HIGH : LOW);
}

void displayStatus(const char* verdict) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("=== STRAWHAT PIRATES ==="));
  display.setCursor(0, 16);
  display.print(F("Material: "));
  display.println(currentMaterial);
  display.setCursor(0, 28);
  display.print(F("Defects:  "));
  display.println(currentDefectCount);
  display.setTextSize(2);
  display.setCursor(0, 44);
  display.println(verdict);
  display.display();
}

void displayBoot() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 10);
  display.println(F("Crack Inspector"));
  display.setCursor(10, 25);
  display.println(F("STRAWHAT PIRATES"));
  display.setCursor(10, 45);
  display.println(F("Waiting for PC..."));
  display.display();
}

void servoSweep(int targetAngle) {
  if (servoPos < targetAngle) {
    for (int pos = servoPos; pos <= targetAngle; pos++) {
      rejectServo.write(pos);
      delay(SERVO_SPEED_MS);
    }
  } else {
    for (int pos = servoPos; pos >= targetAngle; pos--) {
      rejectServo.write(pos);
      delay(SERVO_SPEED_MS);
    }
  }
  servoPos = targetAngle;
}

void fireReject() {
  isRejected = true;
  rejectStartTime = millis();

  servoSweep(SERVO_REJECT);
  digitalWrite(BUZZER_PIN, LOW); 
  relayOn(RELAY_CH1_PIN);
  relayOn(RELAY_CH2_PIN);
  digitalWrite(LED_REJECT_PIN, HIGH);
  digitalWrite(LED_PASS_PIN, LOW);
  displayStatus("REJECT!");
  Serial.println("ACK:REJECT");
}

void clearReject() {
  isRejected = false;

  servoSweep(SERVO_HOME);
  digitalWrite(BUZZER_PIN, HIGH); 
  relayOff(RELAY_CH1_PIN);
  relayOff(RELAY_CH2_PIN);
  digitalWrite(LED_REJECT_PIN, LOW);
  digitalWrite(LED_PASS_PIN, HIGH);
  displayStatus("PASS");
  Serial.println("ACK:PASS");
}

void resetToInitialState() {
  isRejected = false;
  currentMaterial = "---";
  currentDefectCount = 0;

  servoSweep(SERVO_HOME);
  digitalWrite(BUZZER_PIN, HIGH); 
  relayOff(RELAY_CH1_PIN);
  relayOff(RELAY_CH2_PIN);
  digitalWrite(LED_REJECT_PIN, LOW);
  digitalWrite(LED_PASS_PIN, HIGH);
  displayBoot();
  Serial.println("ACK:RESET_INITIAL");
}

void parseStatusCommand(String data) {
  int colonIdx = data.indexOf(':');
  if (colonIdx < 0) return;

  String payload = data.substring(colonIdx + 1);
  int commaIdx = payload.indexOf(',');
  if (commaIdx < 0) return;

  currentMaterial = payload.substring(0, commaIdx);
  currentDefectCount = payload.substring(commaIdx + 1).toInt();

  bool defectFound = (currentDefectCount > 0) || isRejected;
  displayStatus(defectFound ? "REJECT!" : "PASS");
  Serial.print("ACK:STATUS:");
  Serial.print(currentMaterial);
  Serial.print(",");
  Serial.println(currentDefectCount);
}

void processCommand(String cmd) {
  cmd.trim();

  if (cmd == "REJECT") {
    fireReject();
  }
  else if (cmd == "PASS") {
    clearReject();
  }
  else if (cmd.startsWith("STATUS:")) {
    parseStatusCommand(cmd);
  }
  else if (cmd == "PING") {
    Serial.println("ACK:PONG");
  }
  else {
    Serial.print("ERR:UNKNOWN_CMD:");
    Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 Crack Inspector v1.0");

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RELAY_CH1_PIN, OUTPUT);
  pinMode(RELAY_CH2_PIN, OUTPUT);
  pinMode(LED_PASS_PIN, OUTPUT);
  pinMode(LED_REJECT_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, HIGH); 
  relayOff(RELAY_CH1_PIN);
  relayOff(RELAY_CH2_PIN);
  digitalWrite(LED_PASS_PIN, HIGH);
  digitalWrite(LED_REJECT_PIN, LOW);

  rejectServo.attach(SERVO_PIN);
  rejectServo.write(SERVO_HOME);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("ERR:OLED_INIT_FAILED");
  } else {
    displayBoot();
  }

  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }


  if (isRejected && (millis() - rejectStartTime > REJECT_HOLD_MS)) {
    resetToInitialState();
  }
}
