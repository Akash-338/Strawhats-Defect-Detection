/*
 * ESP32 Hardware Reject System with 0.96" SSD1306 OLED Display
 * Board: ESP32 Dev Board
 * 
 * Pinout:
 * - OLED SDA:          GPIO 21 (I2C)
 * - OLED SCL:          GPIO 22 (I2C)
 * - Servo Signal:      GPIO 13 (or GPIO 27)
 * - Red LED / REJECT:  GPIO 14 (or GPIO 4)
 * - Buzzer:            GPIO 26 (or GPIO 32)
 * - Green LED / PASS:  GPIO 27 (or GPIO 2)
 * 
 * Communication: Serial @ 115200 baud
 * Protocol: "REJECT", "PASS", "RESET", "STATUS:<MATERIAL>,<DEFECT_COUNT>"
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>

// ── Pin Definitions ─────────────────────────────────────────────
const int RED_LED_PIN   = 14;
const int BUZZER_PIN    = 26;
const int GREEN_LED_PIN = 27;
const int SERVO_PIN     = 13;

// ── OLED Display Config ────────────────────────────────────────
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT   64
#define OLED_RESET      -1
#define OLED_ADDR       0x3C

#define REJECT_HOLD_MS  2500   // Auto-reset back to initial state after 2.5s

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
Servo rejectServo;

String currentMaterial = "READY";
int    currentDefectCount = 0;
bool   isRejected = false;
bool   hasOled = false;
unsigned long rejectStartTime = 0;
String inputBuffer = "";

void displayHeader() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("=== STRAWHAT PIRATES ==="));
}

void displayBoot() {
  if (!hasOled) return;
  displayHeader();
  display.setCursor(10, 18);
  display.println(F("Crack Inspector v2.0"));
  display.setCursor(10, 34);
  display.println(F("System Ready"));
  display.setCursor(10, 48);
  display.println(F("Waiting for PC..."));
  display.display();
}

void displayStatus(const char* verdict, uint16_t color = SSD1306_WHITE) {
  if (!hasOled) return;
  displayHeader();
  display.setCursor(0, 16);
  display.print(F("Mat: "));
  display.println(currentMaterial);
  display.setCursor(0, 28);
  display.print(F("Defects: "));
  display.println(currentDefectCount);
  
  display.setTextSize(2);
  display.setCursor(0, 44);
  display.println(verdict);
  display.display();
}

void resetState() {
  isRejected = false;
  currentMaterial = "READY";
  currentDefectCount = 0;
  
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  noTone(BUZZER_PIN);
  rejectServo.write(0); // Idle position
  
  displayBoot();
}

void fireReject() {
  isRejected = true;
  rejectStartTime = millis();

  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(RED_LED_PIN, HIGH);
  
  // 1. Loud 2kHz Buzzer Beep (works on both active & passive buzzers)
  tone(BUZZER_PIN, 2000, 600);
  digitalWrite(BUZZER_PIN, HIGH);
  
  // 2. Smooth Step-wise Servo Sweep (0 to 90 degrees)
  for (int pos = 0; pos <= 90; pos += 5) {
    rejectServo.write(pos);
    delay(15);
  }
  displayStatus("REJECT!");
  
  delay(600);
  digitalWrite(BUZZER_PIN, LOW);
  noTone(BUZZER_PIN);
}

void firePass() {
  isRejected = false;
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, HIGH);
  
  // Return servo arm smoothly to 0 degrees
  for (int pos = 90; pos >= 0; pos -= 5) {
    rejectServo.write(pos);
    delay(15);
  }
  rejectServo.write(0);
  displayStatus("PASS");
  
  delay(1500);
  resetState();
}

void parseStatusCommand(String data) {
  int colonIdx = data.indexOf(':');
  if (colonIdx < 0) return;

  String payload = data.substring(colonIdx + 1);
  int commaIdx = payload.indexOf(',');
  if (commaIdx >= 0) {
    currentMaterial = payload.substring(0, commaIdx);
    currentDefectCount = payload.substring(commaIdx + 1).toInt();
  } else {
    currentMaterial = payload;
  }
  displayStatus(isRejected ? "REJECT!" : (currentDefectCount > 0 ? "REJECT!" : "PASS"));
}

void processCommand(String cmd) {
  cmd.trim();
  String cmdUpper = cmd;
  cmdUpper.toUpperCase();

  if (cmdUpper == "REJECT") {
    fireReject();
  }
  else if (cmdUpper == "PASS") {
    firePass();
  }
  else if (cmdUpper.startsWith("STATUS:")) {
    parseStatusCommand(cmd);
  }
  else if (cmdUpper == "RESET") {
    resetState();
  }
}

void setup() {
  Serial.begin(115200);
  
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // ESP32 Servo setup
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  rejectServo.setPeriodHertz(50);
  rejectServo.attach(SERVO_PIN, 1000, 2000); // Standard SG90 pulse width
  
  // Wire I2C OLED setup
  Wire.begin(21, 22); // SDA = 21, SCL = 22
  if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    hasOled = true;
    displayBoot();
  } else {
    hasOled = false;
    Serial.println("WARN:OLED_NOT_FOUND");
  }
  
  resetState();
  Serial.println("STATUS:READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
      }
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // Auto-reset after REJECT_HOLD_MS (2.5 seconds) back to initial WAITING FOR PC state
  if (isRejected && (millis() - rejectStartTime > REJECT_HOLD_MS)) {
    resetState();
  }
}
