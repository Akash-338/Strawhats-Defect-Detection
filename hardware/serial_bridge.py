import logging
import time
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SerialBridge:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.ser = None
        self.connected = False
        self.baudrate = baudrate
        self.port = port
        
        if not HAS_PYSERIAL:
            logging.warning("pyserial not installed. Serial communication disabled.")
            return
            
        self.connect()

    def connect(self) -> bool:
        if self.port is None:
            self.port = self._auto_detect_port()
            
        if self.port:
            for baud in [self.baudrate, 9600, 115200]:
                try:
                    if self.ser and self.ser.is_open:
                        self.ser.close()
                    self.ser = serial.Serial(self.port, baud, timeout=1)
                    # Enable DTR/RTS for STM32 CDC / Virtual COM ports
                    self.ser.dtr = True
                    self.ser.rts = True
                    time.sleep(0.5)  # Allow board USB stack to settle
                    self.baudrate = baud
                    self.connected = True
                    logging.info(f"Connected to Microcontroller (STM32/ESP32/Arduino) on {self.port} at {self.baudrate} baud.")
                    return True
                except Exception as e:
                    logging.warning(f"Failed to connect on {self.port} at {baud} baud: {e}")
                    self.connected = False
            return False
        else:
            logging.warning("No Microcontroller (STM32/ESP32/Arduino) port provided or auto-detected.")
            self.connected = False
            return False

    def _auto_detect_port(self) -> Optional[str]:
        ports = serial.tools.list_ports.comports()
        for port in ports:
            desc = port.description.upper()
            hwid = port.hwid.upper()
            combined = f"{desc} {hwid}"
            if any(k in combined for k in ["STM32", "STM", "STLINK", "ST-LINK", "NUCLEO", "MAPLE", "CDC", "ESP32", "CP210", "CH340", "FTDI", "USB-SERIAL", "ARDUINO"]):
                return port.device
        return None

    def is_connected(self) -> bool:
        return self.connected

    def send(self, command: str) -> bool:
        if not self.connected or self.ser is None or not self.ser.is_open:
            # Auto-reconnect if cable was re-plugged
            if not self.connect():
                logging.debug(f"Simulating serial send (not connected): {command}")
                return False
            
        try:
            cmd_str = f"{command.strip()}\n"
            self.ser.write(cmd_str.encode('utf-8'))
            self.ser.flush()
            logging.info(f"⚡ Serial sent to ESP32: {cmd_str.strip()}")
            return True
        except Exception as e:
            logging.error(f"Error sending to serial: {e}")
            self.connected = False
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            logging.info("Serial port closed.")

if __name__ == "__main__":
    bridge = SerialBridge()
    if bridge.is_connected():
        bridge.send("PASS")
        time.sleep(2)
        bridge.send("REJECT")
        time.sleep(1)
        bridge.send("RESET")
        bridge.close()
