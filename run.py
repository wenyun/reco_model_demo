import tensorflow as tf
import model_fn
import dataset


def pre_steps():
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
    )
    
    strategy = tf.contrib.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        keep_checkpoint_max=3,
        train_distribute=strategy,
    )
   
    predictor = tf.estimator.Estimator(
        model_fn=model_fn.model_fn,
        model_dir='./checkpoint',
        config=run_config,)

    return predictor


def train():
    predictor = pre_steps()
    predictor.train(input_fn=lambda: dataset.build_dataset())


def main():
  train()

if __name__ == '__main__':
    main()
