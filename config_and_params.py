import os

if os.getenv("COLAB_RELEASE_TAG"):
    inColab = True
    inKaggle = not inColab
elif os.getenv("KAGGLE_URL_BASE"):
    inKaggle = True
    inColab = not inKaggle
else: # if you're using a local machine
    inColab = False
    inKaggle = False

host_name_str = os.popen('hostname').read().encode('utf-8')
print(host_name_str)

dataset_select = 'oxiiit'

if inColab:
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
else:
    cwd = os.getcwd()
    if dataset_select == 'oxiiit':
        cwd1 = os.path.join(cwd, f"dataset_1")
    elif dataset_select == 'voc':
        cwd2 = os.path.join(cwd, f"dataset_2")

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
    output_classes = 3
elif dataset_select == 'voc':
    BATCH_SIZE = 30
    BUFFER_SIZE = 1000
    load_from_tfds = False
    img_dir = os.path.join(cwd2, f"JPEGImages")
    mask_dir = os.path.join(cwd2, f"SegmentationClassFinalAug")
    train_txt_path = os.path.join(cwd2, f"DatasetInfo", f"train.txt")
    val_txt_path = os.path.join(cwd2, f"DatasetInfo", f"val.txt")
    input_shape = (128, 128, 3) # input_height, input_width, input_channels
    output_classes = 21

VAL_SUBSPLITS = 5