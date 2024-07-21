import RPi.GPIO as GPIO
import time
m1a = 5
m1b = 6
m1pwm = 13
GPIO.setmode(GPIO.BCM)
GPIO.setup(m1a, GPIO.OUT)
GPIO.setup(m1b, GPIO.OUT)
GPIO.setup(m1pwm, GPIO.OUT)
m1_thr = GPIO.PWM(m1pwm, 1000)
m1_thr.start(0)


def turn(cmd):
    action = 0
    if cmd < 0:
        GPIO.output(m1a, GPIO.HIGH)
        GPIO.output(m1b, GPIO.LOW)
    else:
        GPIO.output(m1a, GPIO.LOW)
        GPIO.output(m1b, GPIO.HIGH)
    if cmd < -100:
        action = 100
    if cmd > -100 and cmd < -30:
        action = abs(cmd)
    if cmd >= -30 and cmd <= 30:
        action = 0
    if cmd > 30:
        action = cmd
    if cmd > 100:
        action = 100
    m1_thr.ChangeDutyCycle(action)

def off():
    GPIO.output(m1a, GPIO.LOW)
    GPIO.output(m1b, GPIO.LOW)

# for i in range(-100, 100, 1):
#     print(i)
#     turn(i)
#     time.sleep(0.1)
# off()
# while(True):
#     turn(-80)
    