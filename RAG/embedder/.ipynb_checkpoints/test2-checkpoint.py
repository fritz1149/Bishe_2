class TT:
    def __init__(self):
        self.a = 1

    __call__ = call

    def call(self):
        print("hello")

e = TT()
e()

    