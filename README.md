This is a framework for anomaly detection mainly based on GAN and AE.
This framework is designed to be **hierarchical and visual**.
This framework can be easily **adjusted to different anomaly detection methods** such as anogan, ganomaly, _etc_.   

Now the GANomaly and f-AnoGAN is accomplished.

For GANomaly, run:

`python train.py dataset yourdataset --nepoch 100 --model ganomaly`

For f-AnoGAN, run:

`python train.py dataset yourdataset --nepoch 100 --nenepoch 100 --model fanogan`

You can load pretrained model by add

`--load_final_weigths` and `--load_final_en_weigths`