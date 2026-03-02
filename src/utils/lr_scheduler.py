class LRScheduler:
    """
    Learning rate scheduler that keeps LR constant for first half of training,
    then linearly decays to zero.
    """

    def __init__(self, optimizer, n_epochs, start_decay_epoch):
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.start_decay_epoch = start_decay_epoch
        self.initial_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.start_decay_epoch:
            lr = self.initial_lr
        else:
            lr = self.initial_lr * (
                1
                - (epoch - self.start_decay_epoch)
                / (self.n_epochs - self.start_decay_epoch)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr
