# SAM-Labelimg
Quick annotation using Segment Anything (SAM) model
This project is based on modification https://github.com/zhouayi/SAM-Tool Library github open source
https://github.com/facebookresearch/segment-anything The library is also an open source project
This project has always been free of charge for data tagging, yolo and other related enthusiasts. All services related to this project, such as use, maintenance, update, dissemination, secondary development (except for secondary development of other bloggers), will not charge any fees. If there is any diversion, such as soliciting attention, adding groups, and for-profit behavior, we will resolutely flee
####Project deployment and use (user)
####1. Download project
###1.1 Source Reference Item 1: https://github.com/zhouayi/SAM-Tool -----Can you do git or not? If you are interested, you can refer to it
####Up main project 1: https://github.com/573951579/Automatic-annotation-too -----Items in video
###1.2 Item 2: https://github.com/facebookresearch/segment-anything -----Sam1 basic library
```bash  
git clone https://github.com/573951579/Automatic-annotation-too.git
renew salt and segment_anything_annotator.py
###
git clone  https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
###If import segment_anything does not report an error in the python environment, the installation succeeds
```
####1.3
Download the 'SAM' model: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth (2.6G)
####1.4
Non project environment missing, such as torch, PyQt5, and other relevant missing and error related tutorials
####2. Place the data in the path '<dataset_path>/images/*' and create an empty folder '<dataset_path>/embeddings`
####The xavier folder contains the test image data set
####Compared with the source project, this project has deleted the steps of manually generating npy files
#####3.2 Run 'generate_onnx. py' to convert the 'pth' file to the 'onnx' model file
```bash
#Cd to the main directory of project 1
python helpers/generate_onnx.py --checkpoint-path sam_vit_h_4b8939.pth --onnx-model-path ./sam_onnx.onnx --orig-im-size 1080 1920
```
-'checkpoint path': the same 'SAM' model path code has added a relative path
-'onnx model path': the saved path of the 'onnx' model obtained
-'orig im size': the size of the image in the data '(height, width)`
[* * Note 1: The 'onnx' model obtained from the code conversion provided does not support dynamic input size, so if the image sizes in your dataset are different, the alternative is to export different 'onnx' models with different 'orig im size' parameters for subsequent use * *]
[* * Note 2: The configuration of each computer and the python environment may be different for the generation of the onnx model, which may not be fully adapted. If you have questions about the generation of the onnx, you can retrieve the relevant answers of the onnx by yourself * *]
####4. The generated 'sam_onnx. onnx' model should be automatically saved to the main directory of project 1, and run 'segment_anything_annotator. py' to mark it
```bash
#Cd to the main directory of project 1
python segment_anything_annotator.py
```
Click the left mouse button at the object location to add the mask, and click the right mouse button to remove the mask.
Other shortcut keys are:
|'Esc': Exit | 'a': Previous picture |'d ': Next picture|
|'r': Reset | 'f': Frame | 'Ctrl+s': Save|
|'Ctrl+z': Undo the last time|
The generated annotation file is in the format of 'coco' and saved in '<dataset_path>/annotations. json'.
Manually export two formats yolov8.txt and yolov11-obb.txt
Yolov8.txt is used for all yolo series basic training
Yolov11-obb.txt is used for basic training of rotation estimation series (obb) after yolov11
If there are questions about the use of the txt dataset during training, feed back to me and fix the bugs in time
####Project library explanation (developer)
./datasets data set folder
./helpers About extract_embeddings folder (attached to the main function for serial operation), sam onnx file generation
./labels txt annotation file generation path
./xavier reference annotation picture
./salt using the tool library
    ./dataset_explorer.py #Read and write json files
    ./display_utils.py #About the drawing of the box in the label screen
    ./editor.py #Main system ui, Annotation interaction and data integration
    ./interface.py #ui design and logic class
    ./json2txt.py #json to txt function
    ./onnx_model.py #onnx model call
    ./utils.py #serves onnx

View the./cocoviewer.py json file
./environment.yaml Recommended environment configuration
./LICENSE use license (not this project)
./README.md Suggestions
## Reference
https://github.com/facebookresearch/segment-anything 
https://github.com/anuragxel/salt
https://github.com/trsvchn/coco-viewer
