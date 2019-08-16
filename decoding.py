r"""Decode from trained T2T models.

Mimic t2t-decoder binary.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os
import re
import string
import sys
import time

import numpy as np
import six

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


def create_hp_and_estimator(
    problem_name, data_dir, checkpoint_path, decode_to_file=None):
  trainer_lib.set_random_seed(FLAGS.random_seed)

  hp = trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(data_dir),
      problem_name=problem_name)

  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.shards = FLAGS.decode_shards
  decode_hp.shard_id = FLAGS.worker_id
  decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
  decode_hp.decode_in_memory = decode_in_memory
  decode_hp.decode_to_file = decode_to_file
  decode_hp.decode_reference = None

  FLAGS.checkpoint_path = checkpoint_path
  estimator = trainer_lib.create_estimator(
      FLAGS.model,
      hp,
      t2t_trainer.create_run_config(hp),
      decode_hparams=decode_hp,
      use_tpu=FLAGS.use_tpu)
  return hp, decode_hp, estimator


def backtranslate_interactively(
    from_problem, to_problem,
    from_data_dir, to_data_dir,
    from_ckpt, to_ckpt):
  
  from_hp, from_decode_hp, from_estimator = create_hp_and_estimator(
      from_problem, from_data_dir, from_ckpt)
  
  to_hp, to_decode_hp, to_estimator = create_hp_and_estimator(
      to_problem, to_data_dir, to_ckpt)

  def interactive_text_input():
    while True:
      if sys.version_info >= (3, 0):
        input_text = input('>>> ')
      else:
        input_text = raw_input('>>> ')

      if input_text == 'q':
        break

      yield input_text

  print('Loading from {} ..'.format(from_ckpt))
  intermediate_lang = decode_interactively(
    from_estimator, interactive_text_input(), 
    from_problem, from_hp, from_decode_hp, from_ckpt)

  print('Loading from {} ..'.format(to_ckpt))
  outputs = decode_interactively(
    to_estimator, intermediate_lang, 
    to_problem, to_hp, to_decode_hp, to_ckpt)

  for output in outputs:
    print('Paraphrased: {}'.format(output.replace('&apos;', "'")))


def decode_interactively(estimator,
                         input_generator,
                         problem_name,
                         hparams,
                         decode_hp,
                         checkpoint_path=None):
  """Compute predictions on entries in filename and write them out."""
  decode_hp.batch_size = 1
  tf.logging.info(
      "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]

  length = getattr(hparams, "length", 0) or hparams.max_length

  def input_fn_gen():
    for line in input_generator:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      else:
        ids = targets_vocab.encode(line)
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      np_ids = np.array(ids, dtype=np.int32)
      yield dict(
          inputs=np_ids.reshape((length, 1, 1))
      )

  def input_fn(params):
    return tf.data.Dataset.from_generator(
      input_fn_gen,
      output_types=dict(
          inputs=tf.int32,
      ),
      output_shapes=dict(
          inputs=(length, 1, 1)
      )
    ).batch(1)

  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  for result in result_iter:
    _, decoded_outputs, _ = decoding.log_decode_results(
        result["inputs"],
        result["outputs"],
        problem_name,
        None,
        inputs_vocab,
        targets_vocab,
        log_results=False,
        skip_eos_postprocess=decode_hp.skip_eos_postprocess)
    yield decoded_outputs


def decode_from_text_file(estimator,
                          problem_name,
                          filename,
                          hparams,
                          decode_hp,
                          decode_to_file=None,
                          checkpoint_path=None):
  """Compute predictions on entries in filename and write them out."""
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]
  filename = decoding._add_shard_to_filename(filename, decode_hp)
  tf.logging.info("Performing decoding from file (%s)." % filename)
  if has_input:
    sorted_inputs, sorted_keys = decoding._get_sorted_inputs(
        filename, decode_hp.delimiter)
  else:
    sorted_inputs = decoding._get_language_modeling_inputs(
        filename, decode_hp.delimiter, repeat=decode_hp.num_decodes)
    sorted_keys = range(len(sorted_inputs))

  # If decode_to_file was provided use it as the output filename without change
  # (except for adding shard_id if using more shards for decoding).
  # Otherwise, use the input filename plus model, hp, problem, beam, alpha.
  decode_filename = decode_to_file if decode_to_file else filename
  if not decode_to_file:
    decode_filename = decoding._decode_filename(decode_filename, problem_name, decode_hp)
  else:
    decode_filename = decoding._add_shard_to_filename(decode_filename, decode_hp)
  tf.logging.info("Writing decodes into %s" % decode_filename)

  # Check for decoding checkpoint.
  decodes = []
  shuffle_file_path = decode_filename + '.shuffle.txt'
  if tf.gfile.Exists(shuffle_file_path):
    with tf.gfile.Open(shuffle_file_path, 'r') as f:
      decodes = [line.strip() for line in f.readlines()]
    tf.logging.info('Read {} sentences from checkpoint.'.format(len(decodes)))

  all_sorted_inputs = sorted_inputs
  # We only need to decode these inputs:
  sorted_inputs = sorted_inputs[len(decodes):]

  # We don't need to waste computation on empty lines:
  num_empty_lines = 0
  while sorted_inputs and sorted_inputs[-1] == '':
    num_empty_lines += 1
    sorted_inputs.pop(-1)

  num_sentences = len(sorted_inputs)
  num_decode_batches = (num_sentences - 1) // decode_hp.batch_size + 1

  if estimator.config.use_tpu:
    length = getattr(hparams, "length", 0) or hparams.max_length
    batch_ids = []
    for line in sorted_inputs:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      else:
        ids = targets_vocab.encode(line)
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      batch_ids.append(ids)
    np_ids = np.array(batch_ids, dtype=np.int32)
    def input_fn(params):
      batch_size = params["batch_size"]
      dataset = tf.data.Dataset.from_tensor_slices({"inputs": np_ids})
      dataset = dataset.map(
          lambda ex: {"inputs": tf.reshape(ex["inputs"], (length, 1, 1))})
      dataset = dataset.batch(batch_size)
      return dataset
  else:
    def input_fn():
      input_gen = decoding._decode_batch_input_fn(
          num_decode_batches, sorted_inputs,
          inputs_vocab, decode_hp.batch_size,
          decode_hp.max_input_size,
          task_id=-1, has_input=has_input)
      gen_fn = decoding.make_input_fn_from_generator(input_gen)
      example = gen_fn()
      return decoding._decode_input_tensor_to_features_dict(example, hparams)
  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  start_time = time.time()
  total_time_per_step = 0
  total_cnt = 0

  def timer(gen):
    while True:
      try:
        start_time = time.time()
        item = next(gen)
        elapsed_time = time.time() - start_time
        yield elapsed_time, item
      except StopIteration:
        break

  writing_mode = 'a' if tf.gfile.Exists(shuffle_file_path) else 'w'
  shuffle_file = tf.gfile.Open(shuffle_file_path, writing_mode)
  count = 0
  for elapsed_time, result in timer(result_iter):
    if decode_hp.return_beams:
      beam_decodes = []
      beam_scores = []
      output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in result:
        if np.isscalar(result["scores"]):
          result["scores"] = result["scores"].reshape(1)
        scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(output_beams):
        tf.logging.info("BEAM %d:" % k)
        score = scores and scores[k]
        _, decoded_outputs, _ = decoding.log_decode_results(
            result["inputs"],
            beam,
            problem_name,
            None,
            inputs_vocab,
            targets_vocab,
            log_results=decode_hp.log_results,
            skip_eos_postprocess=decode_hp.skip_eos_postprocess)
        beam_decodes.append(decoded_outputs)
        if decode_hp.write_beam_scores:
          beam_scores.append(score)
      if decode_hp.write_beam_scores:
        decodes.append("\t".join([
            "\t".join([d, "%.2f" % s])
            for d, s in zip(beam_decodes, beam_scores)
        ]))
      else:
        decodes.append("\t".join(beam_decodes))
    else:
      _, decoded_outputs, _ = decoding.log_decode_results(
          result["inputs"],
          result["outputs"],
          problem_name,
          None,
          inputs_vocab,
          targets_vocab,
          log_results=decode_hp.log_results,
          skip_eos_postprocess=decode_hp.skip_eos_postprocess)
      decodes.append(decoded_outputs)

    # Write decoded text to checkpoint
    new_decode = decodes[-1]
    shuffle_file.write(new_decode + '\n')

    # Flush checkpoint to storage.
    count += 1
    if count % decode_hp.batch_size == 0:
      tf.logging.info('Done {}/{}. Flushing.'.format(
          count, len(sorted_inputs)))
      shuffle_file.flush()
      shuffle_file.close()
      shuffle_file = tf.gfile.Open(shuffle_file_path, 'a')

    total_time_per_step += elapsed_time
    total_cnt += result["outputs"].shape[-1]

  for _ in range(num_empty_lines):
    decodes.append('')
    shuffle_file.write('\n')

  # Write the final output to file.
  outfile = tf.gfile.Open(decode_filename, "w")
  for index in range(len(all_sorted_inputs)):
    outfile.write("%s%s" % (decodes[sorted_keys[index]], 
                            decode_hp.delimiter))
  outfile.flush()
  outfile.close()

  # Close and remove checkpoint.
  shuffle_file.flush()
  shuffle_file.close()
  tf.gfile.Remove(shuffle_file_path)

  # Print some decoding stats.
  duration = time.time() - start_time
  if total_cnt:
    tf.logging.info("Elapsed Time: %5.5f" % duration)
    tf.logging.info("Averaged Single Token Generation Time: %5.7f "
                    "(time %5.7f count %d)" %
                    (total_time_per_step / total_cnt,
                     total_time_per_step, total_cnt))
  if decode_hp.batch_size == 1:
    tf.logging.info("Inference time %.4f seconds "
                    "(Latency = %.4f ms/setences)" %
                    (duration, 1000.0*duration/num_sentences))
  else:
    tf.logging.info("Inference time %.4f seconds "
                    "(Throughput = %.4f sentences/second)" %
                    (duration, num_sentences/duration))


def t2t_decoder(problem_name, data_dir, 
                decode_from_file, decode_to_file,
                checkpoint_path):
  hp, decode_hp, estimator = create_hp_and_estimator(
      problem_name, data_dir, checkpoint_path, decode_to_file)

  decode_from_text_file(
      estimator, problem_name,
      decode_from_file, hp, 
      decode_hp, decode_to_file,
      checkpoint_path=checkpoint_path)