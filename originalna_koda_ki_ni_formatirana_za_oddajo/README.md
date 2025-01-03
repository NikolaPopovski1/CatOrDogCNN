# Convolutional neural network
## Detect if there is a Cat or a Dog in an image

### How to use:
- If you want to train from scratch, you need to set load_model in line 58 to false and set the number of epochs you wish in line 57.
- The model outputs an image that has to be inside the repository, preferably in the folder /test_imgs/, and the name of the file is included in line 118 in main.
- After the image with the label of either a cat or a dog is closed you will get the output of the evaluation of the model in the form of a JSON. Inside this JSON are the model name, model loss, and model accuracy respectively
