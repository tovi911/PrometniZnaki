import os
import subprocess
import time
import sys
import argparse
import requests
import progressbar

FLAGS = None

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_folder = os.path.join(root_folder, "Training", "src", "keras_yolo3")
data_folder = os.path.join(root_folder, "Data")
model_folder = os.path.join(data_folder, "Model_Weights")

if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument(
        "--is_tiny",
        default=False,
        action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False.",
    )

    FLAGS = parser.parse_args()

    weights_file = "PrometniZnaki.weights"
    h5_file = "PrometniZnaki.h5"
    cfg_file = "PrometniZnaki.cfg"

    if os.path.isfile(os.path.join(download_folder, weights_file)):
        
        print(f"Converting {weights_file}\n")

        call_string = f"python {download_folder}/convert.py {cfg_file} {weights_file} {h5_file}"

        subprocess.call(call_string, shell=True, cwd=download_folder)
