"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Contains functions to initialize the model and decode the sequences for the LM Evaluation Harness tasks"""



import datetime
import os
import numpy as np
import pyconfig
import max_utils
from input_pipeline.input_pipeline_interface import create_data_iterator_with_tokenizer
from layers import models
from decode import validate_config, init_decode, prefill_or_load, decode_ar_one_step
import jax
from jax import random
from jax.sharding import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc
import max_logging
import time
from generate_helper import generate
from decode_utils_lmeval import init_model_gen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

Transformer = models.Transformer



def init_gemma(max_target_length=720,max_prefill_predict_length=512):
  """
  The function initializes the model, model variables, tokenizer and random number generator for lm_eval tasks
  Args:
  :param max_target_length: int
    - the maximum length of the generated sequence
  :param max_prefill_predict_length: int
    - the maximum length of the sequence to be pre-filled before decoding
  
  Returns:
  :return: tuple
    - model: layers.models.Transformer
      - the model
    - model_vars: dict
      - dictionary of model parameters
    - tokenizer: Tokenizer
      - the tokenizer
    - rng: jax.random.PRNGKey
      - the random number generator
  """
  import ipdb; ipdb.set_trace()

  time_now = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
  tokenizer_path = 'gs://ishaan-finetuning-experiments/tokenizer.model'
  load_parameters_path = 'gs://ishaan-finetuning-checkpoints/2b/0/default' 
  model_name = 'gemma-2b'
  argv = ['', 
          '/home/ishaanshah/essential_maxtext/maxtext/MaxText/configs/base.yml', 
          f'tokenizer_path={tokenizer_path}', 
          f'load_parameters_path={load_parameters_path}', 
          f'run_name=runner_{time_now}', 
          f'max_prefill_predict_length={max_prefill_predict_length}', 
          f'max_target_length={max_target_length}', 
          'dataset_type=synthetic', 
          'async_checkpointing=false', 
          'attention=dot_product', 
          f'model_name={model_name}',
          f'decode_sampling_temperature=0.1',
          f'decode_sampling_nucleus_p=0.95',
          f'decode_sampling_top_k=0',
          f'decode_sampling_strategy=nucleus']


  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
  
  config = pyconfig.config
  validate_config(config)
  model, model_vars, tokenizer, rng = init_decode(config)
  return model,model_vars,tokenizer,rng

model,model_vars,tokenizer,rng = init_gemma()

def generate_gemma(prompts,num_return_sequences,stopping_criteria=None,max_length=2048,**kwargs):
  """
  The function generates the completions for the given input
  Args:
  :param model
  """
  outputs = []
  if stopping_criteria:
    until = [stopping_criteria]
  else:
    until = None

  
  
  for i in range(num_return_sequences):
    outputs += generate(model,model_vars,tokenizer,rng,[prompts['ids']],max_gen_toks=max_length,until=until)

  return outputs

def generate_batched(model,model_vars,tokenizer,rng,full_prompts,max_gen_toks=2048,until=None):
  outputs = []
  batch_sz = 32
  num_batches = int(np.ceil(len(full_prompts)/batch_sz))

  if until:
    until = [until]*batch_sz
 
  for i in range(num_batches):
      start_time = time.time()
      prompts = full_prompts[i*batch_sz: (i+1)*batch_sz]
      outputs += generate(model,model_vars,tokenizer,rng,prompts,until=until,max_gen_toks=max_gen_toks)
      end_time = time.time()
      print(f'count: {i},  total: {num_batches}, time: {end_time - start_time}')
  return outputs


def generate_samples(prompts,num_return_sequences,stopping_criteria=None,max_length=2048,**kwargs):
  if stopping_criteria:
    until = stopping_criteria
  else:
    until = None
  
  output_per_prompt = [[] for _ in range(len(prompts))]

  for i in range(num_return_sequences):
    outputs = generate_batched(model,model_vars,tokenizer,rng,[p['prompt'] for p in prompts],max_gen_toks=max_length,until=until)
    for j in range(len(prompts)):
      output_per_prompt[j].append(outputs[j])
  return output_per_prompt





if  __name__ == '__main__':
  
  outputs = generate(model,model_vars,tokenizer,rng,['The capital of Turkey is named '])
  print(outputs)
  #model,model_vars,tokenizer,rng = init_gemma()
  