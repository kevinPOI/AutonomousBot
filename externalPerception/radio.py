import sys
import glob
import time
import serial
import serial.tools.list_ports
import random
from pynput import keyboard
import numpy as np
class Radio:
    def __init__(self, baudrate = 115200):
        self.baudrate = baudrate
        self.key_states = {'w': False, 's': False, 'a': False, 'd': False, 'm': False }
        self.mode = 0 #0: manual, 1: auto
        self.listener = None
        self.init_key_listener()
        self.radio = self.find_serial_port()
        self.target_pos = np.array([200,200,0])
    def init_key_listener(self):
        def on_press(key):
            try:
                if key.char in self.key_states:
                    self.key_states[key.char] = True
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char in self.key_states:
                    self.key_states[key.char] = False
                    if key.char == 'm':
                        self.mode = 1 - self.mode
                        print("Curr mode: ", self.mode)
            except AttributeError:
                pass
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
    def send_control(self, control = [0,0]):
        if self.mode == 0:
            self.manual_control()
        else:
            return self.auton_control(control)

    def manual_control(self):
        if self.key_states['w']:
            thr = 0.4
        elif self.key_states['s']:
            thr = -0.4
        else:
            thr = 0

        if self.key_states['a']:
            steer = 0.2
        elif self.key_states['d']:
            steer = -0.2
        else:
            steer = 0

        right_x = int(127 - thr * 127)
        left_y = int(127 - steer * 127)
        joystick_state = [0, left_y, right_x, 0, 0, 0]
        rand_bytes = random.randrange(0, 2**32).to_bytes(4, 'little')
        packed_bytes = rand_bytes + bytes(joystick_state) + rand_bytes
        packed_bytes += bytes([sum(packed_bytes) % 256])
        self.radio.write(packed_bytes)
        print("manual sent: ", joystick_state)

    def auton_control(self, control):
        coef = 0.5
        control *= coef
        thr = control[0]
        steer = control[1]
        right_x = int(127 - thr * 127)
        left_y = int(127 - steer * 127)
        joystick_state = [0, left_y, right_x, 0, 0, 0]
        rand_bytes = random.randrange(0, 2**32).to_bytes(4, 'little')
        packed_bytes = rand_bytes + bytes(joystick_state) + rand_bytes
        packed_bytes += bytes([sum(packed_bytes) % 256])
        self.radio.write(packed_bytes)
        sim_mode = True
        if sim_mode:
            
            if self.key_states['w']:
                self.target_pos[1] -= 5
            elif self.key_states['s']:
                self.target_pos[1] += 5
            else:
                pass

            if self.key_states['a']:
                self.target_pos[0] += 5
            elif self.key_states['d']:
                self.target_pos[0] -= 5
            else:
                pass
            return self.target_pos
        #print("auton sent: ", joystick_state)
    
    def find_serial_port(self):
        if sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.usbserial-*')
        elif sys.platform.startswith('win'):
            ports = filter(lambda x: x.pid == 29987, serial.tools.list_ports.comports())
            ports = list(map(lambda x: x.name, ports))
        else:
            ports = filter(lambda x: x.pid == 29987, serial.tools.list_ports.comports())
            ports = list(map(lambda x: x.name, ports))
            for i in range(len(ports)):
                ports[i] = '/dev/' + ports[i]

        radio = None

        for port in ports:
            try:
                print(port)
                radio = serial.Serial(port, baudrate=self.baudrate)
            except (OSError, serial.SerialException):
                pass

        if radio is None:
            print('Could not find radio! Is the dongle plugged in?')
            sys.exit(1)

        return radio


# def main():
#     radio = find_serial_port()
#     left_x, left_y, right_x, right_y, buttons1, buttons2 = 0, 0, 0, 0, 0, 0

#     thr = 0
#     steer = 0
#     key_states = {'w': False, 's': False, 'a': False, 'd': False}

#     def on_press(key):
#         try:
#             if key.char in key_states:
#                 key_states[key.char] = True
#         except AttributeError:
#             pass

#     def on_release(key):
#         try:
#             if key.char in key_states:
#                 key_states[key.char] = False
#         except AttributeError:
#             pass

#     listener = keyboard.Listener(on_press=on_press, on_release=on_release)
#     listener.start()

#     last_t = time.time()
#     while True:
#         t = time.time()
#         if t - last_t >= 0.03:
#             if key_states['w']:
#                 thr = 0.4
#             elif key_states['s']:
#                 thr = -0.4
#             else:
#                 thr = 0

#             if key_states['a']:
#                 steer = 0.4
#             elif key_states['d']:
#                 steer = -0.4
#             else:
#                 steer = 0

#             right_x = int(127 - thr * 127)
#             left_y = int(127 - steer * 127)
#             joystick_state = [left_x, left_y, right_x, right_y, buttons1, buttons2]

#             last_t = t
#             rand_bytes = random.randrange(0, 2**32).to_bytes(4, 'little')
#             packed_bytes = rand_bytes + bytes(joystick_state) + rand_bytes
#             packed_bytes += bytes([sum(packed_bytes) % 256])
#             radio.write(packed_bytes)


if __name__ == '__main__':
    radio = Radio()
    while True:
        time.sleep(0.05)
        radio.send_control()