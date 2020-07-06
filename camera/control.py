import RPi.GPIO as gpio
import time

# 定义引脚，你怎么接的怎么改
in1 = 7
in2 = 11
in3 = 13
in4 = 15
ENA=35
ENB=37

# 设置GPIO口为BOARD编号规范
gpio.setmode(gpio.BOARD)

# 设置GPIO口为输出
gpio.setup(in1, gpio.OUT)
gpio.setup(in2, gpio.OUT)
gpio.setup(in3, gpio.OUT)
gpio.setup(in4, gpio.OUT)
gpio.setup(ENA, gpio.OUT)
gpio.setup(ENB, gpio.OUT)
pwmA = gpio.PWM(ENA, 500)
pwmB = gpio.PWM(ENB, 500)
pwmA.start(100)
pwmB.start(100)
# 设置输出电平
gpio.output(in1, gpio.HIGH)
gpio.output(in2, gpio.LOW)
gpio.output(in3, gpio.LOW)
gpio.output(in4, gpio.HIGH)

# 秒级延迟
time.sleep(120)
# 释放
pwmA.stop()
pwmB.stop()
gpio.cleanup()
