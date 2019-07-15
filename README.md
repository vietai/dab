# back_translate
Data augmentation with Back Translation

# Requirements

* `tensor2tensor`

# Generate data

For EN -> VI

```
python t2t_datagen.py --data_dir=data/translate_envi_iwslt32k --tmp_dir=tmp/ --problem=translate_envi_iwslt32k
```

For VI -> EN

```
python t2t_datagen.py --data_dir=data/translate_vien_iwslt32k --tmp_dir=tmp/ --problem=translate_vien_iwslt32k
```

# Train

TODO(thtrieu).