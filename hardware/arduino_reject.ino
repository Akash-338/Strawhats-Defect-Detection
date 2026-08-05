#include <Servo.h>

const int redLEDPin = 2;
const int greenLEDPin = 3;
const int buzzerPin = 4;
const int servoPin = 9;

Servo rejectServo;
String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(9600);
  pinMode(redLEDPin, OUTPUT);
  pinMode(greenLEDPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  
  rejectServo.attach(servoPin);
  
  resetState();
  inputString.reserve(200);
}

void loop() {
  if (stringComplete) {
    inputString.trim();
    if (inputString == "REJECT") {
      digitalWrite(redLEDPin, HIGH);
      digitalWrite(buzzerPin, HIGH);
      rejectServo.write(90);
      delay(500);
      digitalWrite(buzzerPin, LOW);
    } 
    else if (inputString == "PASS") {
      digitalWrite(greenLEDPin, HIGH);
      delay(1000);
      digitalWrite(greenLEDPin, LOW);
    }
    else if (inputString == "RESET") {
      resetState();
    }
    
    inputString = "";
    stringComplete = false;
  }
}

void resetState() {
  digitalWrite(redLEDPin, LOW);
  digitalWrite(greenLEDPin, LOW);
  digitalWrite(buzzerPin, LOW);
  rejectServo.write(0);
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
