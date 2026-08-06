/*
 * Arduino / STM32 Hardware Reject System Sketch
 * Board: Arduino Uno / Nano / STM32 (Nucleo / BluePill)
 * 
 * Pinout:
 * - Red LED:   Pin 2
 * - Green LED: Pin 3
 * - Buzzer:    Pin 4
 * - Servo:     Pin 9
 * 
 * Communication: Serial @ 9600 / 115200 baud
 * Protocol: "REJECT", "PASS", "RESET"
 */

#include <Servo.h>

const int redLEDPin   = 2;
const int greenLEDPin = 3;
const int buzzerPin   = 4;
const int servoPin    = 9;

Servo rejectServo;
String inputString = "";
bool stringComplete = false;

void resetState() {
  digitalWrite(redLEDPin, LOW);
  digitalWrite(greenLEDPin, LOW);
  digitalWrite(buzzerPin, LOW);
  rejectServo.write(0); // Idle position at 0 degrees
}

void setup() {
  Serial.begin(115200);
  pinMode(redLEDPin, OUTPUT);
  pinMode(greenLEDPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  
  rejectServo.attach(servoPin, 1000, 2000); // Standard SG90 pulse width
  resetState();
  inputString.reserve(200);
  Serial.println("STATUS:WAITING_FOR_PC");
}

void loop() {
  if (stringComplete) {
    inputString.trim();
    inputString.toUpperCase();
    
    if (inputString == "REJECT") {
      digitalWrite(greenLEDPin, LOW);
      digitalWrite(redLEDPin, HIGH);
      digitalWrite(buzzerPin, HIGH);
      rejectServo.write(90); // Sweep reject arm to 90 degrees
      delay(500);            // 500ms buzzer pulse
      digitalWrite(buzzerPin, LOW);
      delay(1500);           // Hold reject arm & Red LED for 1.5s
      resetState();          // Auto-reset back to WAITING FOR PC
      Serial.println("STATUS:WAITING_FOR_PC");
    } 
    else if (inputString == "PASS") {
      digitalWrite(redLEDPin, LOW);
      digitalWrite(greenLEDPin, HIGH);
      rejectServo.write(0);  // Idle position
      delay(1500);           // Hold Green LED for 1.5s
      resetState();          // Auto-reset back to WAITING FOR PC
      Serial.println("STATUS:WAITING_FOR_PC");
    }
    else if (inputString == "RESET") {
      resetState();
      Serial.println("STATUS:WAITING_FOR_PC");
    }
    
    inputString = "";
    stringComplete = false;
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
