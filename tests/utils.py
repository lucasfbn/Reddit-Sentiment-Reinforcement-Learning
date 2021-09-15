class MockObj(object):
    def __init__(self, **kwargs):  # constructor turns keyword args into attributes
        self.__dict__.update(kwargs)

    def __repr__(self):
        str_ = "MO ("
        for i, (key, val) in enumerate(self.__dict__.items()):
            str_ += f"{key}: {str(val)}"
            if i < len(self.__dict__) - 1:
                str_ += ", "
        str_ += ")"
        return str_
