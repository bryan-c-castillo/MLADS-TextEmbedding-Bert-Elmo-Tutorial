import os

import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

class BertClassifier:
    
    DEFAULT_CONFIG = {
        'bert_url': 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
        'output_dir': 'data/output',
        'data_column': None,
        'label_column': None,
        'max_seq_length': 128,
        'label_values': None,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'num_train_epochs': 3.0,
        'warmup_proportion': 0.1,
        'save_checkpoints_steps': 500,
        'save_summary_steps': 100
    }
    
    def __init__(self, **config):
        self.config = BertClassifier.DEFAULT_CONFIG.copy()
        self.config.update(config)
        
        # Make sure the output directory exists.
        if not os.path.isdir(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])
        
        self._tokenizer = None
        
        if self.config['data_column'] is None:
            raise Exception("data_column parameter is required.")
            
        if self.config['label_column'] is None:
            raise Exception("label_column parameter is required.")

    def predict(self, df):
        input_feautes = self._feature_extractor(df)
        
        predict_input_fn = run_classifier.input_fn_builder(
            features=input_feautes,
            seq_length=self.config['max_seq_length'],
            is_training=False,
            drop_remainder=False)

        # Get the predictions and reformat them.
        # The estimator was setup to return log probability.
        # the formatting will convert back to standard probabilities.
        predictions = self.estimator.predict(predict_input_fn)
        results = []
        for prediction, item in zip(predictions, df[self.config['data_column']]):
            results.append({
                self.config['data_column']: item,
                self.config['label_column']: prediction['labels'],
                'probabilities': dict(zip(self.config['label_values'], np.exp(prediction['probabilities'])))
            })
        return results
        
    def test(self, df):
        test_features = self._feature_extractor(df)
        
        test_input_fn = run_classifier.input_fn_builder(
            features=test_features,
            seq_length=self.config['max_seq_length'],
            is_training=False,
            drop_remainder=False)

        self.test_metrics = self.estimator.evaluate(input_fn=test_input_fn, steps=None)
        
        return self.test_metrics
        
    def train(self, df):
        self._initialize_label_values(df)        
        train_features = self._feature_extractor(df)

        num_train_steps = int(len(train_features) / self.config['batch_size'] * self.config['num_train_epochs'])
        num_warmup_steps = int(num_train_steps * self.config['warmup_proportion'])
        num_train_steps = int(len(train_features) / self.config['batch_size'] * self.config['num_train_epochs'])
        num_warmup_steps = int(num_train_steps * self.config['warmup_proportion'])

        #output directory and number of checkpoint steps to to save
        run_config = tf.estimator.RunConfig(
            model_dir=self.config['output_dir'],
            save_summary_steps=self.config['save_summary_steps'],
            save_checkpoints_steps=self.config['save_checkpoints_steps'])

        #created model function
        model_fn = self._model_fn_builder(
            num_labels=len(self.config['label_values']),
            learning_rate=self.config['learning_rate'],
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": self.config['batch_size']})

        #create an input function for training; drop_remainder=True for TPUs
        train_input_fn = bert.run_classifier.input_fn_builder(
            features=train_features,
            seq_length=self.config['max_seq_length'],
            is_training=True,
            drop_remainder=False)
			
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        
        self.estimator = estimator
        return self.estimator
        
    def tokenizer(self):
        if self._tokenizer is None:
            with tf.Graph().as_default():
                bert_module = hub.Module(self.config['bert_url'])
                tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
                with tf.Session() as sess:
                    vocab_file, do_lower_case = sess.run(
                        [tokenization_info["vocab_file"],
                        tokenization_info["do_lower_case"]])

            self._tokenizer = bert.tokenization.FullTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)

        return self._tokenizer
        
    def _initialize_label_values(self, df):
        """If label values are not set, they are pulled from the data frame."""
        if self.config['label_values'] is None:
            lbl_values = list(np.unique(df[self.config['label_column']].values))
            lbl_values.sort()
            self.config['label_values'] = lbl_values
        
    def _feature_extractor(self, df):
        inputs = df.apply(lambda row: bert.run_classifier.InputExample(
            guid=None,
            text_a = row[self.config['data_column']],
            text_b = None,
            label = row[self.config['label_column']]), axis=1)
            
        tokenizer = self.tokenizer()
        
        return bert.run_classifier.convert_examples_to_features(
            inputs,
            self.config['label_values'],
            self.config['max_seq_length'],
            tokenizer)
    
    def _create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
        bert_module = hub.Module(self.config['bert_url'], trainable=True)
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):

            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, rate=0.1)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (predicted_labels, log_probs)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, predicted_labels, log_probs)

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def _model_fn_builder(self, num_labels, learning_rate, num_train_steps, num_warmup_steps):
        
        # Calculate evaluation metrics. 
        def metric_fn(label_ids, predicted_labels):
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            f1_score = tf.contrib.metrics.f1_score(
                label_ids,
                predicted_labels)
            auc = tf.metrics.auc(
                label_ids,
                predicted_labels)
            recall = tf.metrics.recall(
                label_ids,
                predicted_labels)
            precision = tf.metrics.precision(
                label_ids,
                predicted_labels) 
            true_pos = tf.metrics.true_positives(
                label_ids,
                predicted_labels)
            true_neg = tf.metrics.true_negatives(
                label_ids,
                predicted_labels)   
            false_pos = tf.metrics.false_positives(
                label_ids,
                predicted_labels)  
            false_neg = tf.metrics.false_negatives(
                label_ids,
                predicted_labels)
            return {
                "eval_accuracy": accuracy,
                "f1_score": f1_score,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "true_positives": true_pos,
                "true_negatives": true_neg,
                "false_positives": false_pos,
                "false_negatives": false_neg
            }
               
        def create_model_fn_predicting(features, labels, mode, params):
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            (predicted_labels, log_probs) = self._create_model(
                    True, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
        def create_model_fn_not_predicting(features, labels, mode, params):
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            
            (loss, predicted_labels, log_probs) = self._create_model(
                False, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            
            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)

        def model_fn(features, labels, mode, params):
            if mode == tf.estimator.ModeKeys.PREDICT:
                return create_model_fn_predicting(features, labels, mode, params)
            else:
                return create_model_fn_not_predicting(features, labels, mode, params)
        
        # Return the actual model function in the closure
        return model_fn