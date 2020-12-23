import external.perception_lib.pyperception_lib as pylib
import numpy as np
import time
import threading

class Visualizer(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.kill_received = False

        self.visualizer = pylib.Visualizer()

        self.save_path = None
        self.render = False
        self.rendered_image = None
 
    def run(self):
        self.visualizer.start();

        while not self.kill_received:
            self.visualizer.loop()
            time.sleep(0.1)

            if self.save_path is not None:
                self.visualizer.saveScreenshot(self.save_path)
                self.save_path = None

            if self.render:
                self.rendered_image = self.visualizer.getRenderedImage()
                self.render = False

    def addCloud(self, cloud, size=1):
        self.visualizer.addCloud(cloud, size)

    def swapBuffer(self):
        self.visualizer.swapBuffer()

    def saveScreenshot(self, file):
        self.save_path = file

    def getRenderedImage(self):
        self.render = True
        while self.render:
            time.sleep(0.1)
        return self.rendered_image
