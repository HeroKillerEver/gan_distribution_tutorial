# gan distribution demo
A simple demo of training GAN to generate samples for Gamma distribution 

## Requirements
* [Python 3.5](https://www.python.org/downloads/)
* [TensorFlow 1.6](https://www.tensorflow.org/install/)
* [Numpy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/users/installing.html)
* [Seaborn](https://seaborn.pydata.org/)

## Usage 
```bash
$ python main.py --help
```
This will illustrate all the arguments you could play with the code. 
```bash
A simple demo using gan to generate gamma distributions

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             gpu to use: 0, 1, 2, 3, 4. Default: None
  --lr LR               learning rate. Default: 1e-4
  --iterations ITERATIONS
                        num of iterations. Default: 2000
  --alpha ALPHA         Gamma alpha. Default: 2.
  --scale SCALE         Gamma beta. Default: .5
  --sample_size SAMPLE_SIZE
                        sample size. Default: 100
  --model_save_dir MODEL_SAVE_DIR
                        directory to save model. Default: checkpoints
  --res_save_dir RES_SAVE_DIR
                        directory to save results. Default: results
  --log_save_dir LOG_SAVE_DIR
                        directory to save logs. Default: logs

###########################################################################
```
#### Train the model
Defaultly, the code will train a generator to fit the `Gamma(2, 2)`, you can play with the code by changing the argument `--alpha` and `--beta` to model other Gamma distributions. For example:
```bash
$ python main.py --alpha=1. --beta=2.
```
## Results and Logs
The code will save a video which illustrates the training procedure in the directory `results/` defaultly.

## Author
Haibin Yu/ [@HeroKillerEver](https://github.com/HeroKillerEver)


