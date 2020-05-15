"""Types and methods, common to multiple data sources"""


class Serie:
    """Series"""
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def to_dict(self):
        """Into dict"""
        return {self.name: {"x": self.x, "y": self.y}}
