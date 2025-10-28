class TT:
    def __init__(self):
        self.a = 1

    def call(self):
        print("hello")

    __call__ = call

e = TT()
e()

    