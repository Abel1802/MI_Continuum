We synthesize speech continuum using mini-MI.


## Dependancy
- python 3.6+
- pytorch 1.0+
- pyworld 
- praat-parselmouth

## Folder Structure
  ```
  MI_Continuum/
  |
  |--train.py - main trian to start train
  |
  |--test.py - evaluation of trained model
  |
  |--configs/ - configurations for training
  |  |--base.yaml - base configuration
  |
  |--data_loader/ - anything about data loading
  |  |--data_loader.py
  |
  |--model/ - model archit
  |  |--model.py
  |  |--mi_estimator.py
  |
  |--trainer/ 
  |  |--trainer.py
  |
  |--utils/ - samll utility functions
     |--util.py
  ```

## Preprocess
Our model is trained on [BLCU-SAIT Corpus](https://ieeexplore.ieee.org/abstract/document/7919008)

## Usage

### Training
You can start training by running python train.py. The arguments are listed below.

### Testing
You can inference by running python test.py. The arguments are listed below.

## Contact
If you have any question about the paper or the code, feel free to email me at lzblcu19@gmail.com.
