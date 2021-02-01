

class BlueGraphException(Exception):
    pass


class PGFrameException(BlueGraphException):
    pass


class MetricProcessingException(BlueGraphException):
    pass


class MetricProcessingWarning(UserWarning):
    pass
