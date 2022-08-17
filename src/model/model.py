import os
import shutil
import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

map_name_to_handle = json.load(open("../files/map_name_to_handle.json",'r'))
    
map_model_to_preprocess = json.load(open("../files/map_model_to_preprocess.json",'r'))

def create_model(bert_model_name = 'bert_en_uncased_L-12_H-768_A-12',plot_model=False,model_plot_dir=None):

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
    model = build_classifier_model(tfhub_handle_preprocess,tfhub_handle_encoder)
    if plot_model:
        diag=tf.keras.utils.plot_model(model)
        return model,diag
    return model

def build_classifier_model(tfhub_handle_preprocess,tfhub_handle_encoder):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(3, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)
