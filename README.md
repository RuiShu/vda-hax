# vda-hax

Simple tricks to improve visual domain adaptation for MNIST -> SVHN

A typical implementation of convnets trained on MNIST and tested in SVHN usually yields ~20% accuracy. That being said, it's pretty easy to get to ~40% accuracy on MNIST-> SVHN by applying the following tricks:

1. Apply instance normalization to the input
2. Use batch normalization
3. Use an exponential moving average of the parameter trajectory chosen by your optimizer
4. Add gaussian noise after dropout

I didn't do an extensive ablation test, so it's hard to say which of these contributed the most to the performance increase.

### Run code

Download data first
```bash
python download_svhn.py
python download_mnist.py
```

Run code 
```bash
python run_classifier.py
```
