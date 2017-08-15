class Config():
    def __init__(self):
        self.epochs      = 100
        self.batch_size  = 128
        self.image_size  = 28

        self.save_filename   = 'weights.h5'
        self.callback_period = 5
        self.verbosity       = 1
