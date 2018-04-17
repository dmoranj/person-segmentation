import argparse
import skindataset as sd
import extractskin as es
import trainskinmodel as tsm
import glob
import os

DEFAULT_OUTPUT_PATH='./results'

def generate_description():
    return """"
        This command offers a set of tools for the creation of skin models based on Machine Learning algorithms executed 
        over Computer Vision features.
        
        There are three basic actions that can be executed:
        
        - dataset: this action can be executed over a file or a set of input files in order to extract regions of interest
          to generate a dataset of skin features. The tool shows the selected examples in a window, letting the user
          select the regions of the image that contain skin examples. Once all the regions are selected, pressing C will
          show a new window with just the selected regions. Pressing S will calculate the features for the skin pixels. 
          Once the features have been calculated, the image will be shown again for the user to select non-skin regions.
          This regions of interest can be saved again with the C and S keys. Once all the pixel features have been stored,
          the resulting dataframe will be stored in the selected output path. 
         
        - train: this action takes a set of CSV files created with the previous action to train a Machine Learning model
          (currently Adaboost) that can distinguish between skin and non-skin pixels. The result of this training is a
          binary model stored in the selected location (stored with joblib) along with a report of the training score
          summary.
           
        - extract: the last action can be used to extract a skin mask of a given image using a model trained with the
          previous action. The extracted skin will also be saved along with the mask in the selected output folder 
          (the prefixes "_mask" ad "_skin" will be used for the generated files).
        
    """


def defineParser():
    parser = argparse.ArgumentParser(description=generate_description())
    parser.add_argument('action', type=str, help='Action to execute [dataset, train, extract]')
    parser.add_argument('--inputFolder', dest='inputFolder', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Input directory from where to take the images')
    parser.add_argument('--outputFolder', dest='outputFolder', type=str,
                        help='Name of the output folder')
    parser.add_argument('--modelPath', dest='modelPath', type=str,
                        help='Path to save or load the skin model')

    return parser


def list_from_folder(folder, pattern):
    pathnames = glob.glob(os.path.join(folder, pattern))
    return pathnames


def start():
    args = defineParser().parse_args()

    if args.action == 'dataset':
        sd.process_images(list_from_folder(args.inputFolder, '*.jpg'), args.outputFolder)
    elif args.action == 'train':
        tsm.train_skin_model(list_from_folder(args.inputFolder, '*.csv'), args.modelPath)
    elif args.action == 'extract':
        es.extract_skins(list_from_folder(args.inputFolder, '*.jpg'), args.modelPath, args.outputFolder)
    else:
        print('Action not found\n')


start()
