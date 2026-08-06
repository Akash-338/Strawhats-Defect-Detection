# 🔌 ESP32 Hardware Reject System & OLED Pinout Diagram
**Team Strawhat-Pirates | RVCE Hackathon 2026**

---

## 📌 GPIO Pin Mapping (ESP32 Dev Board)

| Hardware Component | ESP32 GPIO Pin | Power Source | Wire Color / Connection |
|---|---|---|---|
| 🖥️ **0.96" OLED SDA** | **GPIO 21** | I2C Data | SDA Pin on OLED |
| 🖥️ **0.96" OLED SCL** | **GPIO 22** | I2C Clock | SCL Pin on OLED |
| 🖥️ **0.96" OLED VCC** | **3.3V or 5V** | ESP32 Power | VCC Pin on OLED |
| 🔴 **Red LED** | **GPIO 14** | ESP32 3.3V (220Ω Resistor) | Signal $\rightarrow$ Anode, GND $\rightarrow$ Cathode |
| 🔔 **Buzzer** | **GPIO 26** | ESP32 3.3V / 5V | Signal $\rightarrow$ Positive, GND $\rightarrow$ Ground |
| 🟢 **Green LED** | **GPIO 27** | ESP32 3.3V (220Ω Resistor) | Signal $\rightarrow$ Anode, GND $\rightarrow$ Cathode |
| 🦾 **SG90 Servo (Signal)** | **GPIO 13** | ESP32 GPIO | Orange / Yellow Signal Wire |
| 🔋 **SG90 Servo (Power)** | **External 5V VCC** | **External 5V Power Supply** | Red Power Wire |
| ⚡ **Shared Ground** | **ESP32 GND** | **Common Ground Rail** | Black Wire (Connects ESP32 GND + OLED GND + External 5V GND) |

---

## ⚡ Circuit Schematic Diagram

```
                       +-----------------------------------+
                       |         ESP32 Dev Board           |
                       +-----------------------------------+
                       |                                   |
    GPIO 21 (SDA) ------> [ SDA ] ----- (0.96" OLED Screen) |
    GPIO 22 (SCL) ------> [ SCL ] ----- (0.96" OLED Screen) |
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
2. Install libraries via Library Manager:
   - **`Adafruit SSD1306`**
   - **`Adafruit GFX Library`**
   - **`ESP32Servo`**
3. Open `hardware/esp32_oled_reject.ino`.
4. Select board **ESP32 Dev Module** and upload via Micro-USB / USB-C cable.
5. Set Baud rate to **`115200`**.
