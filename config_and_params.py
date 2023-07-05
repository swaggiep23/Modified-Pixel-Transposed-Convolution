import os
import sys

host_name_str = os.popen('hostname').read().encode('utf-8') 
print(host_name_str)

# Check the name of the host when you run it on a local machine, 
# and replace the if and elif condition with the name of your host machine, formatted as a byte string.
# Example: If your host's name is 'abc', set if and elif conditions to -> host_name_str == b'abc\n'

if host_name_str == b'HP\n':
    inColab = False
    inKaggle = False
elif host_name_str != b'HP\n':
    inColab = 'google.colab' in sys.modules
    inKaggle = not inColab

dataset_select = 'oxiiit'

if host_name_str == b'HP\n':
    cwd = os.getcwd()
    if dataset_select == 'oxiiit':
        cwd1 = os.path.join(cwd, f"dataset_1")
    elif dataset_select == 'voc':
        cwd2 = os.path.join(cwd, f"dataset_2")
elif inColab:
    from google.colab import drive
    drive.mount('/content/drive')
    # enter the complete path for the colab notebook.
    cwd = '/content/drive/MyDrive/Colab Notebooks and Jamboards/Colab Notebooks/NNFL/New segmentation code'  
    if dataset_select == 'oxiiit':
        cwd1 = f"{cwd}/dataset_1"
    elif dataset_select == 'voc':
        cwd2 = f"{cwd}/dataset_2"
elif inKaggle:
    cwd_in = '/kaggle/input'
    cwd = '/kaggle/working'
    if dataset_select == 'oxiiit':
        cwd1 = f"{cwd_in}/oxiiit"
    elif dataset_select == 'voc':
        cwd2 = f"{cwd_in}/voc2012mod"

# Dataset parameters
if dataset_select == 'oxiiit':
    BATCH_SIZE = 10
    BUFFER_SIZE = 1000
    load_from_tfds = True
    img_dir = os.path.join(cwd1, f"JPEGImages")
    mask_dir = os.path.join(cwd1, f"SegmentationClass")
    train_txt_path = os.path.join(cwd1, f"DatasetInfo", f"trainval.txt")
    val_txt_path = os.path.join(cwd1, f"DatasetInfo", f"test.txt")
    input_shape = (128, 128, 3) # input_height, input_width, input_channels
    learn_r = 1e-3
    output_classes = 3
elif dataset_select == 'voc':
    BATCH_SIZE = 30
    BUFFER_SIZE = 1000
    load_from_tfds = False
    img_dir = os.path.join(cwd2, f"JPEGImages")
    mask_dir = os.path.join(cwd2, f"SegmentationClassFinal4")
    train_txt_path = os.path.join(cwd2, f"DatasetInfo", f"train3.txt")
    val_txt_path = os.path.join(cwd2, f"DatasetInfo", f"val.txt")
    input_shape = (128, 128, 3) # input_height, input_width, input_channels
    learn_r = 1e-3
    output_classes = 21
