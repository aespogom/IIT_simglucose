from .counterfactual_utils import logger

class EarlyStopper:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            logger.info(f"Early stop counter has increased to {str(self.counter)}")
            if self.counter >= self.patience:
                logger.info("Early stop counter has reached patient!!!!")
                return True
        if validation_loss < 0.001:
            logger.info("Early stop has reached min loss defined as 0.001 !!!!")
            self.counter += self.patience
        return False
