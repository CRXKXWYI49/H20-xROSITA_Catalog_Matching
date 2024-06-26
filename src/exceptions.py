


class DataFrameLengthMismatchError(Exception):
    """Exception raised when pandas DataFrames do not have the same length."""
    def __init__(self, message="DataFrames do not have the same length"):
        self.message = message
        super().__init__(self.message)

