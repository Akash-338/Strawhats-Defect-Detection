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
        
        if not HAS_PYSERIAL:
            logging.warning("pyserial not installed. Serial communication disabled.")
            return
            
        if port is None:
            port = self._auto_detect_port()
            
        if port:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=1)
                time.sleep(2)  # Wait for ESP32/Arduino reset
                self.connected = True
                logging.info(f"Connected to Microcontroller (ESP32/Arduino) on {port} at {baudrate} baud.")
            except Exception as e:
                logging.warning(f"Failed to connect on {port}: {e}")
        else:
            logging.warning("No Microcontroller (ESP32/Arduino) port provided or auto-detected.")

    def _auto_detect_port(self) -> Optional[str]:
        ports = serial.tools.list_ports.comports()
        for port in ports:
            desc = port.description.upper()
            if any(k in desc for k in ["ESP32", "CP210", "CH340", "FTDI", "USB-SERIAL", "ARDUINO"]):
                return port.device
        return None

    def is_connected(self) -> bool:
        return self.connected

    def send(self, command: str) -> bool:
        if not self.connected or self.ser is None:
            logging.debug(f"Simulating serial send (not connected): {command}")
            return False
            
        try:
            cmd_str = f"{command.strip()}\n"
            self.ser.write(cmd_str.encode('utf-8'))
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
