import io
import os
import sys
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class Logger:
    def __init__(self, model_name):
        self.time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.log_dir = os.path.join(f"log/{model_name}", self.time_stamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def save_model_metadata(self, model, input, name, batch_size):
        with open(f"{self.log_dir}/{name}.txt", "w") as f:
            buffer = io.StringIO()
            sys.stdout = buffer
            summary(model, input, batch_size=batch_size)
            sys.stdout = sys.__stdout__
            out = buffer.getvalue()
            f.write(out)
