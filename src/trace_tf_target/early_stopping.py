class EarlyStopping:
    """Early stops training if the monitored metric doesn't improve."""

    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value, model=None):
        score = metric_value

        if self.best_score is None:
            self.best_score = score
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
