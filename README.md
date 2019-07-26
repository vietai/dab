# back_translate

Data augmentation with Back Translation.

# Requirements

* `tensor2tensor`

# Training the two translation models

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

# Back translate from a text file.

We have trained two translation models (`vien` and `envi`) using the `tiny` setting of `tensor2tensor`'s Transformer, and put it on Google Cloud Storage with public access for you to use.

Here is an example of back translating Vietnamese -> English -> Vietnamese from an input text file.

```
python back_translate.py --lang=vi --decode_hparams="beam_size=4,alpha=0.6" --paraphrase_from_file=test_input.vi --paraphrase_to_file=test_output.vi --model=transformer --hparams_set=transformer_tiny
```

# Demo

TODO(thtrieu)