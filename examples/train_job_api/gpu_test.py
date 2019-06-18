import subprocess
def train():
    print(subprocess.check_output(["nvidia-smi"]))
train()
