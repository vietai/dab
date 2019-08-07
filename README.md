# Data augmentation with Back Translation. 

This repository builds on the idea of back translation as a data augmentation method. The idea is simple: translating a sentence in one language to another and then back to the original language. This way one can multiply the size of any NLP dataset. 

<p align="center"> <img src="backtranslate_demo.gif"/> </p>

In this work we focus on Vietnamese datasets since they are typically small in number and size. We present demonstrations of how to use the code in this repository as well as some other free resources through a series of Google Colab.

# Requirements

We make use of the `tensor2tensor` library to build deep neural networks that perform translation.

# Training the two translation models

A prerequisite to performing back-translation is to train two translation models: English to Vietnamese and Vietnamese to English. A demonstration of the following commands to generate data, train and evaluate the models can be found in [this Google Colab](https://colab.research.google.com/drive/1Bx5HfxbmXnMK7kBLHlmGyhVhQVVrDI0p).

## Generate data (tfrecords)

For English -> Vietnamese

```
python t2t_datagen.py --data_dir=data/translate_envi_iwslt32k --tmp_dir=tmp/ --problem=translate_envi_iwslt32k
```

For Vietnamese -> English

```
python t2t_datagen.py --data_dir=data/translate_vien_iwslt32k --tmp_dir=tmp/ --problem=translate_vien_iwslt32k
```

## Train

Some examples to train your translation models with the Transformer architecture:

For English -> Vietnamese

```
python t2t_trainer.py --data_dir=path/to/tfrecords --problem=translate_envi_iwslt32k --hparams_set=transformer_base --model=transformer --output_dir=path/to/ckpt/dir
```

For Vietnamese -> English

```
python t2t_trainer.py --data_dir=path/to/tfrecords --problem=translate_vien_iwslt32k --hparams_set=transformer_base --model=transformer --output_dir=path/to/ckpt/dir
```

## Analyse the trained models

Once you finished training and evaluating the models, you can certainly play around with them a bit. For example, you might want to run some interactive translation and/or visualize the attention masks for your inputs of choice. This is demonstrated in [this Google Colab](https://colab.research.google.com/drive/1WnQRAFof7P7tvJn_zyb1bIVvz3nBU2QM).

# Back translate from a text file.

We have trained two translation models (`vien` and `envi`) using the `tiny` setting of `tensor2tensor`'s Transformer, and put it on Google Cloud Storage with public access for you to use.

Here is an example of back translating Vietnamese -> English -> Vietnamese from an input text file.

```
python back_translate.py --lang=vi --decode_hparams="beam_size=4,alpha=0.6" --paraphrase_from_file=test_input.vi --paraphrase_to_file=test_output.vi --model=transformer --hparams_set=transformer_tiny
```

For a demonstration of augmenting real datasets with back-translation and obtaining actual gains in accuracy, checkout [this Google Colab](https://colab.research.google.com/drive/1_I0KvFlHFyBcTRT3Bfx9BGLJcIHGJNrG)!


## BibTex


```
@article{trieu19backtranslate,
  author  = {Trieu H. Trinh and Thang Le and Phat Hoang and Minh{-}Thang Luong},
  title   = {Back Translation as Data Augmentation Tutorial},
  journal = {https://github.com/vietai/back_translate},
  year    = {2019},
}
```