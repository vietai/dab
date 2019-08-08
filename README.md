# :sparkles: Data Augmentation with Back Translation :sparkles:

This repository builds on the idea of back translation [1] as a data augmentation method [2, 3]. The idea is simple: translating a sentence in one language to another and then back to the original language. This way one can multiply the size of any NLP dataset. An example using our code is shown below:

<p align="center"> <img src="gif/envien_demo.gif"/> </p>

## Google Colaboratory Tutorials :ant:

We have prepared here a series of Google Colabs Notebooks to walk you through how to use our code in very practical contexts and with the most accessible writing styles. For example, you will be shown how to make use of free computational and free storage resources to replicate all of our results. Here are the Colabs:

:notebook: [Interactive Back Translation](https://colab.research.google.com/github/vietai/back_translate/blob/master/colab/Interactive_Back_Translation.ipynb): A minimal Colab for you to play with our final results. We use this colab to generate the GIF you saw above.

:notebook: [Training Translation Models](https://colab.research.google.com/github/vietai/back_translate/blob/master/colab/T2T_translate_vi%3C_%3Een_tiny_tpu.ipynb): How to connect to GPU/TPU and Google Drive/Cloud storage, download training/testing data from the internet and train/evaluate your models. We use the IWSLT'15 dataset, off-the-shelf implementation from `tensor2tensor` with its `transformer_tiny` setting and obtain the following result:


<table align="center">
<thead>
<tr>
<th></th>
<th>BLEU score</th>
</tr>
</thead>
<tbody>
<tr>
<td>English to Vietnamese</td>
<td>28.65</td>
</tr>
<tr>
<td>Vietnamese to English</td>
<td>27.78</td>
</tr>
</tbody>
</table>


As of this writing, the result above is already competitive with the state-of-the-art (29.6 BLEU score) [4] without using semi-supervised or multi-task learning. More importantly, this result is good enough to be useful for the purpose of this project! Check the following Colabs to see how we make use of these translation models to improve result on a real dataset:

:notebook: [Analyse your Translation Models](https://colab.research.google.com/github/vietai/back_translate/blob/master/colab/Vietnamese_Backtranslation_Model_Analysis.ipynb): Play with and visualize the trained models attention.


<p align="center"> <img src="gif/attn_viz.gif"/> </p>

:notebook: [Use Translation Models to Augment An NLP Dataset](https://colab.research.google.com/github/vietai/back_translate/blob/master/colab/Sentiment_Analysis_%2B_Back_translation.ipynb): See an example of how to augment a small NLP dataset in Vietnamese using your translation models and obtain real gains on test set. On a Vietnamese Sentiment Analysis dataset with only 10K examples, we use back-translation to double the training set size and obtain an improvement of near 2.5\% in absolute accuracy:


<table align="center">
<thead>
<tr>
<th>Original set</th>
<th>Augmented by Back Translation</th>
</tr>
</thead>
<tbody>
<tr>
<td>83.48 %</td>
<td><strong>85.91 %</strong></td>
</tr>
</tbody>
</table>


Here is another GIF demo with a Vietnamese sentence, for fun

<p align="center"> <img src="gif/vienvi_demo.gif"/> </p>

## How to contribute? :thinking:

Initially we trained only Vietnamese-English and English-Vietnamese models for back-translation. The code in this repository, however, can work with any other pair of languages. We therefore invite :sparkling_heart: :hand: and **pull requests** from you on:

:seedling: More and/or better translation models.

:seedling: More and/or better translation data or monolingual data.

:seedling: Code to make our code even easier to use. Including tests ([Travis](https://github.com/marketplace/travis-ci), [CodeCov](https://github.com/codecov)).

:seedling: Texts/Illustrations to make our documentation even easier to understand.

We will be working on a more detailed guideline for contribution.

## BibTex :honeybee: :honeybee:

If you make use of code/resources provided in this project, please cite using the following BibTex:

```
@article{trieu19backtranslate,
  author  = {Trieu H. Trinh and Thang Le and Phat Hoang and Minh{-}Thang Luong},
  title   = {Back Translation as Data Augmentation Tutorial},
  journal = {https://github.com/vietai/back_translate},
  year    = {2019},
}
```

## References :cherry_blossom:

[1] Sennrich, Rico, Barry Haddow, and Alexandra Birch. "Improving neural machine translation models with monolingual data.", ACL 2016.

[2] Edunov, Sergey, et al. "Understanding back-translation at scale.",  EMNLP 2018.

[3] Xie, Qizhe, et al. "Unsupervised data augmentation." arXiv preprint arXiv:1904.12848 (2019).

[4] Clark, Kevin, et al. "Semi-supervised sequence modeling with cross-view training.", EMNLP 2018.


# Appendix: Command Syntaxes.

The remaining of this `README` is for those who cannot have access to our Colab Notebooks and/or only need a quick reference to the command syntax of our code.

## Requirements

We make use of the `tensor2tensor` library to build deep neural networks that perform translation.

## Training the two translation models

A prerequisite to performing back-translation is to train two translation models: English to Vietnamese and Vietnamese to English. A demonstration of the following commands to generate data, train and evaluate the models can be found in [this Google Colab](https://colab.research.google.com/github/vietai/back_translate/blob/master/colab/T2T_translate_vi%3C_%3Een_tiny_tpu.ipynb).

### Generate data (tfrecords)

For English -> Vietnamese

```
python t2t_datagen.py --data_dir=data/translate_envi_iwslt32k --tmp_dir=tmp/ --problem=translate_envi_iwslt32k
```

For Vietnamese -> English

```
python t2t_datagen.py --data_dir=data/translate_vien_iwslt32k --tmp_dir=tmp/ --problem=translate_vien_iwslt32k
```

### Train

Some examples to train your translation models with the Transformer architecture:

For English -> Vietnamese

```
python t2t_trainer.py --data_dir=path/to/tfrecords --problem=translate_envi_iwslt32k --hparams_set=transformer_base --model=transformer --output_dir=path/to/ckpt/dir
```

For Vietnamese -> English

```
python t2t_trainer.py --data_dir=path/to/tfrecords --problem=translate_vien_iwslt32k --hparams_set=transformer_base --model=transformer --output_dir=path/to/ckpt/dir
```

### Analyse the trained models

Once you finished training and evaluating the models, you can certainly play around with them a bit. For example, you might want to run some interactive translation and/or visualize the attention masks for your inputs of choice. This is demonstrated in [this Google Colab](https://colab.research.google.com/github/vietai/back_translate/blob/master/colab/Vietnamese_Backtranslation_Model_Analysis.ipynb).

## Back translate from a text file.

We have trained two translation models (`vien` and `envi`) using the `tiny` setting of `tensor2tensor`'s Transformer, and put it on Google Cloud Storage with public access for you to use.

Here is an example of back translating Vietnamese -> English -> Vietnamese from an input text file.

```
python back_translate.py --lang=vi --decode_hparams="beam_size=4,alpha=0.6" --paraphrase_from_file=test_input.vi --paraphrase_to_file=test_output.vi --model=transformer --hparams_set=transformer_tiny
```

Add `--backtraslate_interactively` to back-translate interactively from your terminal. Alternatively, you can also check out [this Colab](https://colab.research.google.com/github/vietai/back_translate/blob/master/colabs/Interactive_Back_Translation.ipynb).

For a demonstration of augmenting real datasets with back-translation and obtaining actual gains in accuracy, checkout [this Google Colab](https://colab.research.google.com/github/vietai/back_translate/blob/master/colab/Sentiment_Analysis_%2B_Back_translation.ipynb)!