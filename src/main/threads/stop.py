from pathlib import Path
from threading import Event, Thread


class StopThread(Thread):
    def __init__(self, file: Path):
        super().__init__(name="Stop")
        self.file = file
        self.start()

    def run(self):
        while self.file.exists() and not STOP.wait(1):
            continue
        STOP.set()


STOP = Event()
