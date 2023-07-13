import tensorflow as tf
import matplotlib.pyplot as plt
import os
from config_and_params import *


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_list, model, model_name, cwd):
        super().__init__()
        # self.validation_data = None
        # self.model = None
        # self._chief_worker_only = None
        # self._supports_tf_logs = False
        self.image_list = image_list
        self.model = model
        self.model_name = model_name
        self.cwd = cwd

    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)
        show_predictions(epoch, self.model_name, self.image_list, self.model,
                         self.cwd, mode=1, num=3)
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
                       "image_list": self.image_list,
                       "model": self.model,
                       "model_name": self.model_name,
                       "cwd": self.cwd                    
                       })

        return config


def lr_scheduler(epoch, lr, lr_init, max_epochs):
#     slow_iters = 0.1*max_epochs
    power = 0.9
#     if epoch+1 <= slow_iters:
#         return lr_init*((epoch+1)/slow_iters)
#     else:
    return lr_init*(1-epoch/max_epochs)**power


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(epoch, model_name, image_list, model, cwd,
                     mode=0, dataset=None, num=1, WIDTH=8, HEIGHT=8):
    """
    *****************
    *** Arguments ***
    *****************
    epoch: (Integer) 
        Only active for 'Mode 1'. For other modes, set it to the string that is to be displayed or None.
    model_name: (String) 
        The name of the model, as it is to be saved. (format - model_number)
    image_list: (List)
        The list of images, that are to be converted and displayed.
    model: (tf.model)
        The tensorflow model that is to be used to generate predictions.
    mode: (Integer)
        Sets the display type to be used. 
            * Mode '0' for images and masks only. (Preview the dataset before training)
            * Mode '1' for images, masks and predictions. (Training dataset)
            * Mode '2' for images, masks and predictions. (Validation dataset)
            * Mode '3' for images and predictions only. (Test dataset)
    dataset: (tf.dataset)
        The dataset to preview, if any.
    num: (Integer) 
        The number of images to be displayed.
    """
    if dataset:
        imgs_masks_and_preds = []
        if mode == 3:
            for image in dataset.take(num):
                imgs_masks_and_preds.append(image[0])
                pred_mask = model.predict(image)
                imgs_masks_and_preds.append(create_mask(pred_mask))  
        else:
            for image, mask in dataset.take(num):
                imgs_masks_and_preds.append(image[0])
                imgs_masks_and_preds.append(mask[0])
                if mode == 1 or mode == 2:
                    pred_mask = model.predict(image)
                    imgs_masks_and_preds.append(create_mask(pred_mask))
    else:
        imgs_masks_and_preds = []
        step = 1 if mode == 3 else 2
        for index in range(0, len(image_list), step):
            imgs_masks_and_preds.append(image_list[index])
            if mode != 3:
                imgs_masks_and_preds.append(image_list[index+1])
            pred_mask = model.predict(image_list[index][tf.newaxis, ...])
            imgs_masks_and_preds.append(create_mask(pred_mask))
    display(epoch, model_name, imgs_masks_and_preds, num,
            cwd, mode, WIDTH, HEIGHT)
    
            
def display(epoch, model_name, display_list, num_subplts, 
            cwd, mode=0, WIDTH=8, HEIGHT=8):
    """
    *****************
    *** Arguments ***
    *****************
    epoch: (Integer)
        Only active for 'Mode 1'. For other modes, set it to the string that is to be displayed or None.
    model_name: (String) 
        The name of the model, as it is to be saved. (format - model_number)
    display_list: (List of image data) 
        The list of images to be converted and displayed.
    num_subplts: (Integer)
        The number of subplots in the image.
    mode: (Integer)
        Sets the display type to be used. 
            * Mode '0' for images and masks only. (Preview the dataset before training)
            * Mode '1' for images, masks and predictions. (Training dataset)
            * Mode '2' for images, masks and predictions. (Validation dataset)
            * Mode '3' for images and predictions only. (Test dataset)
    """

    title_0 = ['Input Image', 'True Mask']
    title_train = ['Input Image', 'True Mask', 'Predicted Mask']
    title_test = ['Input Image', 'Predicted Mask']
    num_cols = int(len(display_list)/num_subplts)

    f, axes = plt.subplots(num_subplts, num_cols, facecolor='black', figsize=(WIDTH,HEIGHT))

    for i in range(len(display_list)):
        ax = plt.subplot(num_subplts, num_cols, i+1, facecolor='black')

        ax.set_facecolor('black')
        ax.xaxis.label.set_color('white')     
        ax.yaxis.label.set_color('white')       
        ax.tick_params(axis='x', colors='white')    
        ax.tick_params(axis='y', colors='white')  
        ax.spines['left'].set_color('white')        
        ax.spines['top'].set_color('white')         
        ax.spines['bottom'].set_color('white') 
        ax.spines['right'].set_color('white') 

        if mode == 0:
            ax.set_title(title_0[i%num_cols], color='white')
        elif mode == 1 or mode == 2:
            ax.set_title(title_train[i%num_cols], color='white')
        elif mode == 3:
            ax.set_title(title_test[i%num_cols], color='white')
        plt.imshow(tf.keras.utils.array_to_img(display_list[i])) 
        plt.axis('off')

    if type(epoch) is bool or type(epoch) is str:
        f.suptitle(f'{epoch}', color='white')
        if mode == 3:
            fig_path = os.path.join(cwd, f'results', f'test_sample_results', f'{model_name}',
                                    f'{epoch}.png')
            try:
                plt.savefig(fig_path)
            except:
                os.makedirs(os.path.dirname(fig_path))
                plt.savefig(fig_path)
        elif mode == 2:
            fig_path = os.path.join(cwd, f'results', f'validation_sample_results', f'{model_name}', 
                                    f'{epoch}.png')
            try:
                plt.savefig(fig_path)
            except:
                os.makedirs(os.path.dirname(fig_path))
                plt.savefig(fig_path)
        elif mode == 1:
            fig_path = os.path.join(cwd, f'results', f'training_sample_results', f'{model_name}',
                                    f'{epoch}.png')
            try:
                plt.savefig(fig_path)
            except:
                os.makedirs(os.path.dirname(fig_path))
                plt.savefig(fig_path)
    else:
        f.suptitle(f'Epoch {epoch+1}', color='white')
        if mode == 1:
            fig_path = os.path.join(cwd, f'results', f'training_sample_results', f'{model_name}', 
                                    f'Epoch - {epoch+1}.png')
            try:
                plt.savefig(fig_path)
            except:
                os.makedirs(os.path.dirname(fig_path))
                plt.savefig(fig_path)
    # plt.show()
    plt.close()


def display_test(pred_batches, PRED_LENGTH, loaded_model, model_num):
    imgs_masks_and_preds = []
    for image in pred_batches.take(PRED_LENGTH):
        pred_mask = loaded_model.predict(image)
        imgs_masks_and_preds.append((image[0], create_mask(pred_mask)))

    for index, (image, pred) in enumerate(imgs_masks_and_preds):
        display(f'Test Result {index+1} - Model {model_num}', f'model_{model_num}', [image, pred], 1,
                cwd, mode=3, WIDTH=8, HEIGHT=8)

    
            