class Logger:
    def __init__(self, summary_writer, metric_summary_freq=100, start_step=0):
        self.summary_writer = summary_writer

        self.total_steps = start_step
        self.metric_summary_freq = metric_summary_freq

        self.running_loss = {}

    def print_training_status(self, mode='train'):
        print(f'Step: {self.total_steps:06d} \t total: {(self.running_loss["total_loss"] / self.metric_summary_freq):.8f}')
        for k in self.running_loss:
            self.summary_writer.add_scalar(mode + '/' + k,
                                           self.running_loss[k] / self.metric_summary_freq, self.total_steps)
            self.running_loss[k] = 0.0
        self.summary_writer.flush()

    def push(self, metrics, mode='train'):
        self.total_steps += 1

        # Running mean
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]

        # Loggin on tensorboard
        if self.total_steps % self.metric_summary_freq == 0:
            self.print_training_status(mode)
            self.running_loss = {}

    def write_dict(self, results, step=None):
        log_step = step if step is not None else self.total_steps
        for key in results:
            self.summary_writer.add_scalar(key, results[key], log_step)
        self.summary_writer.flush()

    def close(self):
        self.summary_writer.close()

    def add_image_summary(self, img_dict, step=None):
        if not step:
            step = self.total_steps
        for k, v in img_dict.items():
            self.summary_writer.add_image(k, v, step)
        self.summary_writer.flush()
