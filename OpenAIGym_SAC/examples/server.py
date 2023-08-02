from crypt import methods
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from examples.sunrise import experiment
from threading import Thread

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

@app.route('/training', methods=['POST'])
@cross_origin()
def train():
    input = request.json
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=int(input['epoch']),
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=int(input['batch_size']),
            save_frequency=int(input['save_frequency']),
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-3,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        num_ensemble=int(input['num_ensemble']),
        num_layer=int(input['num_layer']),
        seed=int(input['seed']),
        ber_mean=0.5,
        env=input['env'],
        inference_type=1,
        temperature=int(input['temperature']),
        log_dir="",
    )
    print(input)
    thread = Thread(target=experiment, args=(variant,))
    thread.start()
    return Response(status=200)
    # experiment(variant)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)