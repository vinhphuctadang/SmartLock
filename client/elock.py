import os 
is_gpio_env = False 
if os.environ.get('GPIO', None):
	is_gpio_env = True 

if is_gpio_env:
	import RPi.GPIO as GPIO

import time

def setLock(lockState):
	print('Going to set lock to state:', lockState)
	if not is_gpio_env:
		print('Warning: Not GPIO environment, going to fake result')
		return 
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)

	GPIO.setup(21, GPIO.OUT)
	lockState = {True: GPIO.HIGH, False: GPIO.LOW}[lockState]
	GPIO.output(21,lockState)

# test GPIO
print('Checking lock health')
# setLock(False)
# time.sleep(1)
# setLock(True)
# time.sleep(1)
# setLock(False)