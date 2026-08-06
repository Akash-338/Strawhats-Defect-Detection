
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
  rejectServo.write(0); 
}

void setup() {
  Serial.begin(115200);
  pinMode(redLEDPin, OUTPUT);
  pinMode(greenLEDPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  
  rejectServo.attach(servoPin, 1000, 2000); 
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
      rejectServo.write(90); 
      delay(500);            
      digitalWrite(buzzerPin, LOW);
      delay(1500);           
      resetState();          
      Serial.println("STATUS:WAITING_FOR_PC");
    } 
    else if (inputString == "PASS") {
      digitalWrite(redLEDPin, LOW);
      digitalWrite(greenLEDPin, HIGH);
      rejectServo.write(0);  
      delay(1500);           
      resetState();          
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
