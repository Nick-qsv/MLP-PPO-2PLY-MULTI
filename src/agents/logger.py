from tensorboardX import SummaryWriter
import botocore
import io
import os
from datetime import datetime
import tensorboardX


class S3Logger:
    def __init__(self, log_dir=None, s3_bucket_name=None, s3_log_prefix="logs/"):
        # Save the original RecordWriter
        self.original_RecordWriter = tensorboardX.record_writer.RecordWriter

        # Define the custom S3RecordWriter
        from tensorboardX.record_writer import S3RecordWriter

        class MyS3RecordWriter(S3RecordWriter):
            def __init__(self, logdir, *args, **kwargs):
                super(MyS3RecordWriter, self).__init__(logdir, *args, **kwargs)

            def flush(self):
                self.buffer.seek(0)
                try:
                    self.s3.upload_fileobj(self.buffer, self.bucket, self.path)
                except botocore.exceptions.ClientError as e:
                    print(f"S3 upload failed: {e}")
                    # Optionally, log the error or take other action
                except Exception as e:
                    print(f"Unexpected exception during S3 upload: {e}")
                finally:
                    self.buffer.close()
                    self.buffer = io.BytesIO()

        # Define a custom RecordWriter function that uses your custom S3RecordWriter
        def MyRecordWriter(logdir, filename_suffix=""):
            if logdir.startswith("s3://"):
                return MyS3RecordWriter(logdir)
            else:
                # Use the original RecordWriter for local directories
                return self.original_RecordWriter(logdir, filename_suffix)

        # Monkey-patch the RecordWriter in tensorboardX
        tensorboardX.record_writer.RecordWriter = MyRecordWriter

        # Initialize the TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if s3_bucket_name:
            # If S3 bucket is provided, set log_dir to S3 path
            self.log_dir = f"s3://{s3_bucket_name}/{s3_log_prefix}{timestamp}/"
        elif log_dir is None:
            self.log_dir = os.path.join("runs", f"backgammon_ppo_{timestamp}")
        else:
            # Append timestamp to the provided log_dir to make it unique
            self.log_dir = os.path.join(log_dir, f"{timestamp}")
        self.writer = SummaryWriter(logdir=self.log_dir)
        print(f"Logging to TensorBoard at {self.log_dir}")

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins="tensorflow"):
        self.writer.add_histogram(tag, values, global_step, bins)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def close(self):
        self.writer.close()
        # Restore the original RecordWriter
        tensorboardX.record_writer.RecordWriter = self.original_RecordWriter
