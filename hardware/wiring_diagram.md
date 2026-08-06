# 🔌 ESP32 Hardware Reject System & OLED Pinout Diagram
**Team Strawhat-Pirates | RVCE Hackathon 2026**

---

## 📌 GPIO Pin Mapping (ESP32 Dev Board)

| Hardware Component | ESP32 GPIO Pin | Description |
|---|---|---|
| 🖥️ **0.96" OLED SDA** | **GPIO 21** | I2C Data (SDA) |
| 🖥️ **0.96" OLED SCL** | **GPIO 22** | I2C Clock (SCL) |
| 🦾 **SG90 Servo (Signal)** | **GPIO 27** | Servo PWM Signal Wire |
| 🔔 **Active-Low Buzzer** | **GPIO 32** | Active-Low Buzzer (LOW = Sound, HIGH = Silent) |
| ⚡ **Relay Channel 1** | **GPIO 33** | Active-Low Relay Channel 1 |
| ⚡ **Relay Channel 2** | **GPIO 25** | Active-Low Relay Channel 2 |
| 🟢 **Green LED (PASS)** | **GPIO 2** | PASS Indicator LED |
| 🔴 **Red LED (REJECT)** | **GPIO 4** | REJECT Indicator LED |
| ⚡ **Ground / VCC** | **GND / 5V** | Common Ground & 5V VCC |

---

## ⚡ Serial Command Protocol (115200 Baud)

- `REJECT` $\rightarrow$ Sweeps servo to 90°, turns Active-Low Buzzer ON, triggers Relays, lights Red LED, displays `REJECT!` on OLED, holds for 3s, then auto-resets back to initial state (`STATUS:WAITING_FOR_PC`).
- `PASS` $\rightarrow$ Sweeps servo to 0° (home), lights Green LED, displays `PASS` on OLED.
- `STATUS:<MATERIAL>,<DEFECT_COUNT>` $\rightarrow$ Updates OLED display with material name and defect count.
- `PING` $\rightarrow$ Returns `ACK:PONG`.
