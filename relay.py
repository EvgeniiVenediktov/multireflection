from time import sleep
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(2,GPIO.OUT)
GPIO.output(2,GPIO.LOW) # activate motor
sleep(3)
GPIO.output(2,GPIO.HIGH) # deactivate motor
sleep(2)
GPIO.cleanup()
