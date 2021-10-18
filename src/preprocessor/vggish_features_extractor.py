# We need soundfile to load audio data
import soundfile as sf
# And the openmic-vggish preprocessor
import openmic.vggish

import os

from Attention import *

DATA_ROOT = '.'
audio, rate = sf.read(os.path.join(DATA_ROOT, 'Chopin_Nocturne_in_C_sharp_Minor_(No20).wav'))

time_points, features = openmic.vggish.waveform_to_features(audio, rate)

log_dir = './log/ISMIR2019/'
model_type = 'attention'
args_id = 'decisionlevelsingleattention_128_3_0.6_lr0.0005_noannealing_res_seed_0'

base_path = os.path.join(log_dir, model_type)
writer_path = os.path.join(base_path, args_id)

model_path = os.path.join('.', 'best_val_loss.pth')
emb_layers = 3
hidden_size = 128
dropout_rate = 0.6
model = DecisionLevelSingleAttention(
                freq_bins=128,
                classes_num=20,
                emb_layers=emb_layers,
                hidden_units=hidden_size,
                drop_rate=dropout_rate)
model.load_state_dict(torch.load(model_path))
model.eval()