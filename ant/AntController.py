import haiku as hk
import jax
import time
import pickle
from WalktCycleConfigParser import WalkCycle
from ArduinoSerial import ArduinoSerial

def net(current_position):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(8)
    ])
    return (mlp(current_position) * 2) - 1

rng = jax.random.PRNGKey(42)

wc = WalkCycle()
input_dataset = wc.get_training_data()
current_position, label = next(input_dataset)

net_t = hk.transform(net)
net_t.init(rng, current_position)
params = pickle.load( open( "params.p", "rb" ))

evaluate = jax.jit(net_t.apply)

arduino_controller = ArduinoSerial('/dev/ttyUSB1')
try:
    while True:
        current_position = evaluate(params, None, current_position)
        print(WalkCycle.frame_to_command(current_position))
        arduino_controller.send(WalkCycle.frame_to_command(current_position))

        time.sleep(0.05)
finally:
    arduino_controller.send_centre_command()
    arduino_controller.close_ports()
    print('ports closed')