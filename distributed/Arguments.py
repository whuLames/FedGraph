
class Arguments(object):
    def __init__(self, args):
        self.args = args
        for key in args.keys():
            value = args[key]
            self.add_arg(key, value)


    def add_arg(self, key, value):
        setattr(self, key, value)
    
    def print_arg(self):
        for key in self.args.keys():
            print('key: {}  value: {}'.format(key, getattr(self, key)))