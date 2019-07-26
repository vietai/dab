"""Back Translation to augment a dataset."""

from __future__ import print_function
from __future__ import division

from tensor2tensor.data_generators import translate_envi
from tensor2tensor.utils import registry


# End-of-sentence marker.
EOS = translate_envi.EOS

# For English-Vietnamese the IWSLT'15 corpus
# from https://nlp.stanford.edu/projects/nmt/ is used.
# The original dataset has 133K parallel sentences.
_VIEN_TRAIN_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-vi/raw/master/data/train-en-vi.tgz",  # pylint: disable=line-too-long
    ("train.vi", "train.en")
]]

# For development 1,553 parallel sentences are used.
_VIEN_TEST_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz",  # pylint: disable=line-too-long
    ("tst2012.vi", "tst2012.en")
]]


@registry.register_problem
class TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


# Here we need to define some dummy problems that do not have any decoding hook.
# In T2T, the original problems will attempt to write to the pretrained checkpoint
# directory (Google Cloud Storage). In general it is not possible to do so
# as an anonymous user having read-only access to the Storage.


@registry.register_problem
class TranslateVienIwslt32kDecode(TranslateVienIwslt32k):
  """Problem spec for IWSLT'15 Vietnamese to English translation."""

  @property
  def decode_hooks(self):
    return []

  def dataset_filename(self):
    return 'translate_vien_iwslt32k'
  


@registry.register_problem
class TranslateEnviIwslt32kDecode(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 English to Vietnamese translation."""

  @property
  def decode_hooks(self):
    return []

  def dataset_filename(self):
    return 'translate_envi_iwslt32k'