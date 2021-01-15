import RPi.GPIO as GPIO
# from time import sleep

def setLock(lockState):
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)

	GPIO.setup(21, GPIO.OUT)
	lockState = {True: GPIO.HIGH, False: GPIO.LOW}[lockState]
	GPIO.output(21,lockState)

setLock(False)

