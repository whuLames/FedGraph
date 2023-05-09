class clientSP:
    def __init__(self, args, trainer) -> None:
        self.args = args

        self.trainer = trainer

        self.round_now_index = 0

        self.rounds = args.comm_round
        
    def train(self):
        return self.trainer.trainLP()
        # return self.trainer.train()

    