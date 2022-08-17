import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt
from ..model import create_model

def train(path,return_hsitory=True):
    model=create_model()
    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        os.path.join(path,'train'),
        batch_size=batch_size,
        seed=seed)

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        os.path.join(path,'val'),
        batch_size=batch_size,
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        os.path.join(path,'test'),
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    epochs = 7
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')
    checkpoint_path = os.path.join(path,"training","cp.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)

    
    history = model.fit(x=train_ds,
                                validation_data=val_ds,
                                epochs=epochs,callbacks=[cp_callback])
    if return_hsitory:
        return model,history
    return 
    

if __name__=='__main__':
    path = '/data/'
    train (path)