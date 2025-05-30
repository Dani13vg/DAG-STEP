# STEP - Towards Structured Scene-Text Spotting

This repository contains the code and data for the modifications done to the [STEP repository](https://github.com/CVC-DAG/STEP) to perform Scene-Text Spotting using more complex and flexible regex representations that allow for easier regex prompting of the model and enhance the performance. To know more about STEP, read the following publication [STEP - Towards Structured Scene-Text Spotting](https://arxiv.org/abs/2309.02356)

![STEP](figures/STEP.png)

## Changes

With respect to the original code in [STEP repository](https://github.com/CVC-DAG/STEP), there are some changes in the following files:

- **adet/data/dataset_mapper.py**: This file was modified to change the way queries are created and stored. Queries are already created and stored along with the rest of the data in the JSON files, so this file simply stores them as tensors of token IDs instead of multihot vectors.

- **adet/data/datasets/text.py**: Line 197 to add the storage of regexes since now the JSON files read have a new keyword 'regex'.

- **adet/data/builtin.py**: Change the paths of the train and validation data files.

- **adet/data/modeling.py**: Process the new query format and perform the correct padding in the `preprocess_queries` function.

- **adet/modeling/step/models.py**: Changes in the model architecture by replacing the MLP encoder of the multihot vectors with the new Transformer encoder that receives the token ids of the regexes. The weights for the encoder are loaded in this file and the forward function is modified to deal with the new format of the data.

- **inference/demo.py**: The script now contains a list of regexes to test, which can be modified and written using regex syntax. You can run the script to test the model on a single image or a folder with images and the model will perform on each of the images with each of the regexes in the list. Results are stored in a given output directory and the weights of the models must be given as well.

## New scripts

The following scripts include new functions used by the modified ones or to create the data JSON files used for this experiment:

- **adet/utils/tokenizer.py**: Contains the functions to load or train the tokenizer and the function to decode the tokens.

- **reformat_data.py**: Used to create the regexes for the annotations of the previous dataset. The script uses LLM's from GROQ (an API Key is required and you can easily get one in [GROQ API KEYS](https://console.groq.com/keys)) to generate the 3 regexes for each string in the annotations and ensures no regex is repeated and they all compile and match the given string. Note that the models from GROQ have a request limitation of 1k requests per model. The script automatically changes between models when a model has reached ist limit and does simple augmentations to the regex to ensure diversity.

## Running the Code

### Code and Environment Setup

Use the following commands to clone and create the environment with conda:

```bash
git clone https://github.com/CVC-DAG/DAG-STEP.git STEP
cd STEP
conda create -n STEP python=3.8 -y
conda activate STEP
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 cudatoolkit-dev=11.3 -c pytorch -c conda-forge
python -m pip install scipy numba
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
python setup.py build develop
```

### Datasets

Our proposed approach uses [HierText-based](https://github.com/google-research-datasets/hiertext) training 
and validation splits. The training and validation images can be downloaded using
the [AWS CLI interface](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html):

````bash
mkdir datasets
mkdir datasets/hiertext
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/train.tgz datasets/hiertext
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/validation.tgz datasets/hiertext
tar -xzvf datasets/hiertext/train.tgz -C datasets/hiertext/
tar -xzvf datasets/hiertext/validation.tgz -C datasets/hiertext/
````

Our pipeline uses custom training and validation ground truths. The ground truth files can be downloaded 
with the following script:

````bash
wget http://datasets.cvc.uab.cat/STEP/structured_ht.zip -P datasets/hiertext
unzip datasets/hiertext/structured_ht.zip -d datasets/hiertext
````

Finally, our proposed test set can be downloaded with:

````bash
wget http://datasets.cvc.uab.cat/STEP/structured_test.zip -P datasets
unzip datasets/structured_test.zip -d datasets
````

The license plate images are sourced from the [UFPR-ALPR](https://github.com/raysonlaroca/ufpr-alpr-dataset)
dataset. The images of this dataset are licensed for non-commercial use and you need to request access 
to the authors (instructions are included in the linked repository).
The images we used are the first frames of each one of the sequences. Supposing that you have
been given access to the UFPR-ALPR dataset, you can download, unzip and copy these frames with:

```bash
wget https://redacted/UFPR-ALPR.zip # change it to the actual download link
unzip UFPR-ALPR.zip
cp UFPR-ALPR\ dataset/*/*/*\[01\].png datasets/structured_test/
# cp UFPR-ALPR\ dataset/**/*\[01\].png datasets/structured_test/ # for zsh
# rm -r UFPR-ALPR\ dataset/  # optionally
```

### Test Dataset Format

The dataset format follows the [labelme](https://github.com/labelmeai/labelme/tree/main) annotation
format. The "label" field of every annotation is its type/class of code (UIC, BIC, tonnage, etc.). The
field "transcription" contains the transcription of the instance. The following table specifies the 
label of every type of code and its regular expression:

| Class  | Regular Expression | Label |
| ------------- | ------------- | ------------- |
| BIC  | \\[A-Za-z]{4}\\s\\d{6}\\s\\d  | bic |
| UIC  | \\d{2}\\s?\\d{2}\\s?\\d{4}\\s?\\d{3}\\-\\d  | uic |
| TARE  | \\d{2}[.]?\\d{3}\\s?(?i)kg  | tare |
| Phone Num.  | \\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4} | phone |
| Tonnage  | \\d{2}[.]?\\d?[t] | tonnage |
| License Plate  | \\[A-Z]{3}\\s\\d{4} | lp |

### Model Weights

The TESTR pretrained on HierText (which are used to initialise the model) and STEP's
final weights can be downloaded with:
```bash
mkdir ckp
wget http://datasets.cvc.uab.es/STEP/TESTR_pretrain_ht_final.pth -P ckp
wget http://datasets.cvc.uab.es/STEP/STEPv1_final.pth -P ckp
```

They should be placed under the ``ckp`` directory, although you can place them anywhere else, but you 
should change the arguments of the example script calls below.

## Running the Model

The model can be trained with the following script (needs the TESTR pretrained weights linked above
in the ``ckp`` folder):

```bash
python tools/train_net.py --config-file configs/STEP/hiertext/STEP_R_50_Polygon.yaml --num-gpus 2
```

You can check the configs/STEP/hiertext/STEP_R_50_Polygon.yaml file to modify some hyperparameters of the model and change the weights of the `Regex Encoder`, or leave an empty string if you don't want to use pretrained weights.

To run the validation script:

```bash
python inference/eval.py --config-file configs/STEP/hiertext/STEP_R_50_Polygon.yaml --opts MODEL.WEIGHTS ckp/STEPv1_final.pth MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.3
 ```

Finally, to run the test script on our proposed test dataset:

```bash
python inference/test.py --config-file configs/STEP/hiertext/STEP_R_50_Polygon.yaml --opts MODEL.WEIGHTS ckp/STEPv1_final.pth MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.3
```

## Create Your Own Queries

You can define your own queries (regexes) in the demo.py script using regex syntax. These queries are given to the model to perform over the given images. You can run this script with:

```bash
python inference/demo.py --config-file configs/STEP/hiertext/STEP_R_50_Polygon.yaml --input-folder DAG-STEP/datasets/hiertext/validation --output </path/to/outputs> --opts MODEL.WEIGHTS </path/to/model/model.pth> MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.2
```

You can save the result in a folder with the flag ```--output <OUTPUT_PATH>```.

## License

This repository is released under the Apache License 2.0. Check the [LICENSE](LICENSE) file, dawg.

## Acknowledgements

We thanks [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) training and inference framework 
and the authors of [TESTR](https://github.com/mlpc-ucsd/TESTR) for their code and work.
