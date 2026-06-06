import fire
from train import test

if __name__ == "__main__":
    # This forces the script to run the test pipeline on the CPU
    fire.Fire(test)