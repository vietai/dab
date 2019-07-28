"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_decoder
from tensor2tensor.models import transformer

import decoding
import problems
import tensorflow as tf
import os


registry = problems.registry

tf.flags.DEFINE_string(
    'vien_problem', 
    'translate_vien_iwslt32k', 
    'Problem name for Vietnamese to English translation.')
tf.flags.DEFINE_string(
    'envi_problem', 
    'translate_envi_iwslt32k', 
    'Problem name for English to Vietnamese translation.')
tf.flags.DEFINE_string(
    'vien_data_dir', 
    'gs://vien-translation/data/translate_vien_iwslt32k', 
    'Data directory for Vietnamese to English translation.')
tf.flags.DEFINE_string(
    'envi_data_dir', 
    'gs://vien-translation/data/translate_envi_iwslt32k', 
    'Data directory for English to Vietnamese translation.')
tf.flags.DEFINE_string(
    'vien_ckpt', 
    'gs://vien-translation/checkpoints/translate_vien_iwslt32k_tiny/avg/', 
    'Pretrain checkpoint directory for Vietnamese to English translation.')
tf.flags.DEFINE_string(
    'envi_ckpt', 
    'gs://vien-translation/checkpoints/translate_envi_iwslt32k_tiny/avg/', 
    'Pretrain checkpoint directory for English to Vietnamese translation.')
tf.flags.DEFINE_string(
    'paraphrase_from_file', 
    'test_input.vi', 
    'Input text file to paraphrase.')
tf.flags.DEFINE_string(
    'paraphrase_to_file', 
    'test_output.vi', 
    'Output text file to paraphrase.')

tf.flags.DEFINE_string('lang', 'vi', 'Language of the input text file (vi or en).')

FLAGS = tf.flags.FLAGS
  


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  # Set up some decoding flags depending on the input text language.
  if FLAGS.lang == 'vi':
    from_data_dir, to_data_dir = FLAGS.vien_data_dir, FLAGS.envi_data_dir
    from_problem, to_problem = FLAGS.vien_problem, FLAGS.envi_problem
    from_ckpt, to_ckpt = FLAGS.vien_ckpt, FLAGS.envi_ckpt
    proxy_lang = 'en'
  elif FLAGS.lang == 'en':
    from_data_dir, to_data_dir = FLAGS.envi_data_dir, FLAGS.vien_data_dir
    from_problem, to_problem = FLAGS.envi_problem, FLAGS.vien_problem
    from_ckpt, to_ckpt = FLAGS.envi_ckpt, FLAGS.vien_ckpt
    proxy_lang = 'vi'
  else:
    raise ValueError('Not supported language: {}'.format(lang))

  # Convert directory into checkpoints
  if tf.gfile.IsDirectory(from_ckpt):
    from_ckpt = tf.train.latest_checkpoint(from_ckpt)
  if tf.gfile.IsDirectory(to_ckpt):
    to_ckpt = tf.train.latest_checkpoint(to_ckpt)

  # For back translation, we need a temporary file in the other language
  # before back-translating into the source language.
  tmp_file = os.path.join(
      '{}.tmp.{}.txt'.format(FLAGS.paraphrase_from_file, proxy_lang)
  )

  # Step 1: Translating from source language to the other language.
  if not tf.gfile.Exists(tmp_file):
    decoding.t2t_decoder(from_problem, from_data_dir,
                         FLAGS.paraphrase_from_file, tmp_file,
                         from_ckpt)

  # Step 2: Translating from the other language (tmp_file) to source.
  decoding.t2t_decoder(to_problem, to_data_dir,
                       tmp_file, FLAGS.paraphrase_to_file,
                       to_ckpt)
