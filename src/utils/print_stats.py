import shutil


def print_disk_usage():
    total, used, free = shutil.disk_usage("/")
    print("*******Disk Space Usage:*******")
    print(f"Total: {total / (1024**3):.2f} GB")
    print(f"Used: {used / (1024**3):.2f} GB")
    print(f"Free: {free / (1024**3):.2f} GB")


def print_shared_memory_usage():
    total, used, free = shutil.disk_usage("/dev/shm")
    print("*******Shared Memory Usage:*******")
    print(f"Total: {total / (1024**3):.2f} GB")
    print(f"Used: {used / (1024**3):.2f} GB")
    print(f"Free: {free / (1024**3):.2f} GB")
