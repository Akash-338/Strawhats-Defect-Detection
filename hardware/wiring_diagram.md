# 🔌 ESP32 Hardware Reject System Pinout & Wiring Diagram
**Team SafePath | RVCE Hackathon 2026**

---

## 📌 GPIO Pin Mapping (ESP32 Dev Board)

| Hardware Component | ESP32 GPIO Pin | Power Source | Wire Color / Connection |
|---|---|---|---|
| 🔴 **Red LED** | **GPIO 14** | ESP32 3.3V (220Ω Resistor) | Signal $\rightarrow$ Anode, GND $\rightarrow$ Cathode |
| 🔔 **Buzzer** | **GPIO 26** | ESP32 3.3V / 5V | Signal $\rightarrow$ Positive, GND $\rightarrow$ Ground |
| 🟢 **Green LED** | **GPIO 27** | ESP32 3.3V (220Ω Resistor) | Signal $\rightarrow$ Anode, GND $\rightarrow$ Cathode |
| 🦾 **SG90 Servo (Signal)** | **GPIO 13** | ESP32 GPIO | Orange / Yellow Signal Wire |
| 🔋 **SG90 Servo (Power)** | **External 5V VCC** | **External 5V Power Supply** | Red Power Wire |
| ⚡ **Shared Ground** | **ESP32 GND** | **Common Ground Rail** | Black Wire (Connects ESP32 GND + External 5V GND) |

---

## ⚡ Circuit Schematic Diagram

```
                       +-----------------------------------+
                       |         ESP32 Dev Board           |
                       +-----------------------------------+
                       |                                   |
    GPIO 14 -----------> [ 220Ω ] ----> (Red LED) -------->|
                       |                                   |---> Common Ground Rail (GND)
    GPIO 26 --------------------------> (Buzzer) --------->|
                       |                                   |
    GPIO 27 -----------> [ 220Ω ] ----> (Green LED) ------>|
                       |                                   |
    GPIO 13 -----------> [ Signal ] --- (SG90 Servo)        |
                       +-----------------------------------+
                                             |
                                 +-----------+-----------+
                                 |  External 5V Power    |
                                 |  +5V -> Servo Power   |
                                 |  GND -> Common Rail   |
                                 +-----------------------+
```

---

## ⚙️ Microcontroller Firmware Installation

1. Open Arduino IDE or PlatformIO.
2. Install the **`ESP32Servo`** library via Library Manager.
3. Open `hardware/esp32_reject.ino`.
4. Select board **ESP32 Dev Module** and upload via USB-C / Micro-USB cable.
5. Set Baud rate to **`115200`** in Python (`inference/realtime.py`).
