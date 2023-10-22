
import datetime
import time
from dateutil import parser

class StreamEvent:
    event: str
    time_stamp: str
    def __init__(self, event, time_stamp):
        self.event = event
        self.time_stamp = time_stamp

class SensoryStream:
    def __init__(self):
        self.events:[StreamEvent] = []

    def _time_stamp(self):
        utc_time = datetime.datetime.utcnow()
        return utc_time.isoformat()

    def _time_since(self, time_stamp):
        current_utc_time = datetime.datetime.utcnow()
        past_time = parser.parse(time_stamp)
        time_difference = current_utc_time - past_time
        return time_difference.total_seconds()

    def append_event(self, event):
        now = self._time_stamp()
        stream_event = StreamEvent(event=event, time_stamp=now)
        self.events.append(stream_event)

    def append_assistant_message(self, message):
        self.append_event(f"Assistant: {message}")

    def append_user_message(self, message):
        self.append_event(f"User: {message}")

    def _pretty_print_time_since(self, time_stamp):
        time_since = self._time_since(time_stamp)
        if time_since < 1:
            return "just now"
        if time_since < 60:
            return f"{int(time_since)} seconds ago"
        elif time_since < 3600:
            return f"{int(time_since / 60)} minutes ago"
        elif time_since < 86400:
            return f"{int(time_since / 3600)} hours ago"
        else:
            return f"{int(time_since / 86400)} days ago"

    def pritty_print(self):
        lines:str = []
        for event in self.events:
            time_since = self._pretty_print_time_since(event.time_stamp)
            lines.append(f"{event.event} - {time_since}")
        _str = "\n".join(reversed(lines))
        return _str

if __name__ == "__main__":
    stream = SesoryStream()
    stream.append_event("new user enters the chat")
    print("-----")
    print(stream.pritty_print())
    time.sleep(3)
    stream.append_assistant_message("Hello")
    print("-----")
    print(stream.pritty_print())
    time.sleep(1)
    stream.append_user_message("Hi")
    print("-----")
    print(stream.pritty_print())
    time.sleep(1)
    stream.append_assistant_message("How are you?")
    print("-----")
    print(stream.pritty_print())
    time.sleep(2)
    stream.append_user_message("Good")
    print("-----")
    print(stream.pritty_print())
    print("")
    