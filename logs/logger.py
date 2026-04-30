import os
import sys
from datetime import datetime

class _Tee:
  def __init__(self, *streams):
    self._streams = streams

  def write(self, data):
    for s in self._streams:
      s.write(data)

  def flush(self):
    for s in self._streams:
        s.flush()

  def fileno(self):
    return self._streams[0].fileno()

  def isatty(self):
    return self._streams[0].isatty()


def setup_logging(script_name: str) -> None:
  logs_dir = os.path.dirname(os.path.abspath(__file__))
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_path = os.path.join(logs_dir, f"{script_name}_{timestamp}.log")
  log_file = open(log_path, "w", encoding="utf-8", buffering=1)
  sys.stdout = _Tee(sys.__stdout__, log_file)
  sys.stderr = _Tee(sys.__stderr__, log_file)
  print(f"Logging to {log_path}")
