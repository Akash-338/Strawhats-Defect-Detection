/*
 * ESP32 Hardware Reject System Sketch
 * Board: ESP32 Dev Board
 * 
 * Pinout (matches team diagram):
 * - Red LED:   GPIO 14
 * - Buzzer:    GPIO 26
 * - Green LED: GPIO 27
 * - SG90 Servo Signal: GPIO 13 (5V external supply, shared ground)
 * 
 * Communication: Serial @ 115200 baud
 * Commands: "REJECT", "PASS", "RESET"
 */

#include <ESP32Servo.h>

const int RED_LED_PIN   = 14;
const int BUZZER_PIN    = 26;
const int GREEN_LED_PIN = 27;
const int SERVO_PIN     = 13;

Servo rejectServo;
String inputBuffer = "";

void resetState() {
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  rejectServo.write(0); // Idle position
}

void setup() {
  Serial.begin(115200);
  
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // Allow allocation of all timers for ESP32 Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  rejectServo.setPeriodHertz(50);    // Standard 50Hz servo
  rejectServo.attach(SERVO_PIN, 500, 2400); // SG90 pulse width min/max us
  
  resetState();
  Serial.println("ESP32_REJECT_SYSTEM_READY");
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
}

void processCommand(String cmd) {
  cmd.toUpperCase();
  
  if (cmd == "REJECT") {
    digitalWrite(GREEN_LED_PIN, LOW);
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    rejectServo.write(90); // Sweep reject arm to 90 degrees
    delay(500);
    digitalWrite(BUZZER_PIN, LOW); // Turn off buzzer sound after 500ms
  } 
  else if (cmd == "PASS") {
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, HIGH);
    rejectServo.write(0);
    delay(1000);
    digitalWrite(GREEN_LED_PIN, LOW);
  }
  else if (cmd == "RESET") {
    resetState();
  }
}
