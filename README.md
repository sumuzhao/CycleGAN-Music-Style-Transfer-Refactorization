# CycleGAN-Music-Style-Transfer-Refactorization
## Symbolic Music Genre Transfer with CycleGAN - Refactorization

Since the project - CycleGAN-Music-Style-Transfer was published, quite a lot people were interested in it. Due to lacking coding experiences, however, there were some annoying problems like following which confused people a lot: 
- Code is not that readable. 
- Code structure is bad. 
- Comments are not enough. 
- Data preprocessing part is not easy to use. 
- Module import problem. 

And there were also some requests like:
- Provide the pretrained model. 
- Provide the dataset which could be fed into the model directly. 
- Network improvement, such as introducing WGAN or WGAN-GP.
- Extend Single track to Multiple tracks

I'm quite busy with my study and other stuff, here sorry for my irresponsiveness to these problems and requests previously. Thus, I want to refactor the previous project in high-level APIs such as TensorFlow 2.0 with Keras, Pytorch and MXNet with Gluon, which makes it easy for you to read and play. In this repository, I mainly focus on improving the code quality. 

Still, I cannot guarantee the codes make everyone happy and reproduce the same results as the original ones. I'll try my best. Feel free to raise issues or pull requests with comments. LIVE AND LEARN! 💪

## Datasets

All the data we used to generate the audio samples on Youtube and for the evaluation in the paper can be downloaded here https://goo.gl/ZK8wLW. I recommend the data set in https://drive.google.com/file/d/1zyN4IEM8LbDHIMSwoiwB6wRSgFyz7MEH/view?usp=sharing, which can be used directly. They are the same as the dataset above. 

**Note**: For those interested in details, please go to [CycleGAN-Music-Style-Transfer](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer) or refer to the paper [Symbolic Music Genre Transfer with CycleGAN](https://arxiv.org/pdf/1809.07575.pdf). 
