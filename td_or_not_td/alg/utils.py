import time
import numpy as np
from os.path import join as p_join


class TimeMeasurement:
    def __init__(self, size=1):
        """Class for multiple time measurements.

        Args:
            size (int): Total amount of time measurements.
        """
        self.size = size
        self.time_total = np.zeros([self.size], dtype=np.float64)
        self.time_start = np.empty([self.size], dtype=np.float64)

    def start(self, number=0):
        """Start time measurement.

        Args:
            number (int): Measurement id.
        """
        self.time_start[number] = time.time()

    def end(self, number=0):
        """End time measurement.

        Args:
            number (int): Measurement id.
        """
        self.time_total[number] += time.time() - self.time_start[number]

    def get(self, number=0):
        """Get time measurement.

        Args:
            number (int): Measurement id.
        """
        return self.time_total[number]

    def reset(self):
        """Reset all time measurements."""
        self.time_total[...] = 0.


class PeriodicEvent:
    def __init__(self, occurrence, first_event_at_zero_time=False):
        """Event manager for training with uneven time-step increment

        Args:
            occurrence (int): is_event will be true once every "occurrence" steps.
            first_event_at_zero_time (bool): If true, event also happens at zeroth time-step.
        """
        self.occurrence = occurrence
        self.last_event_time = 0
        self.event = first_event_at_zero_time

    def update_time(self, current_time):
        """
        Args:
            current_time (int): current time-step.
        """
        if (current_time - self.last_event_time) >= self.occurrence:
            self.last_event_time = (current_time // self.occurrence) * self.occurrence
            self.event = True

    def is_event(self):
        """
        returns (bool): True if "occurrence" time-steps passed since last event.
        """
        if self.event:
            self.event = False
            return True
        else:
            return False


class NetworkSaver:
    """ Saves and loads Tensorflow models."""
    def __init__(self, session, saver, main_path):
        self.session = session
        self.saver = saver
        self.main_path = main_path

        self.save_id = 0

    def save(self):
        save_path = self.saver.save(
            self.session, p_join(self.main_path, "model_{}.ckpt".format(self.save_id))
        )
        print("Model saved in: {}".format(save_path))
        self.save_id += 1

    def load(self, save_id):
        self.saver.restore(self.session, p_join(self.main_path, "model_{}.ckpt".format(save_id)))
