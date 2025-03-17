from time import sleep
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(2,GPIO.OUT)
GPIO.output(2,GPIO.LOW)
sleep(2)
GPIO.output(2,GPIO.HIGH)
sleep(2)
GPIO.cleanup()
