import time


class MotorsModule:
    def __init__(self):
        import board
        from adafruit_motor import servo
        from adafruit_pca9685 import PCA9685
        i2c = board.I2C()
        pca = PCA9685(i2c)
        pca.frequency = 50

        # init all servos
        self.servos = []
        for i in range(16):
            self.servos.append( servo.Servo(pca.channels[i]))

    def test(self, id):
        # We sleep in the loops to give the servo time to move into position.
        for i in range(180):
            self.servos[id].angle = i
            time.sleep(0.03)
        for i in range(180):
            self.servos[id].angle = 180 - i
            time.sleep(0.03)

    def test_all(self):
        for id in [0,3,6,9,12,15]:
            self.test(id)

    def SetServoTargetDegrees(self, servo_i, degrees):
        self.servos[servo_i].angle = degrees
        time.sleep(1)

    def open_claw(self):
        self.SetServoTargetDegrees(5, 30)
        self.SetServoTargetDegrees(6, 30)

    def close_claw(self):
        self.SetServoTargetDegrees(5, 0)
        self.SetServoTargetDegrees(6, 0)

if __name__ == '__main__':
    motors = MotorsModule()
    #motors.open_claw()
    #motors.close_claw()
    #motors.test()
    motors.test_all()

