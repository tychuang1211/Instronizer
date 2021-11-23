# We need soundfile to load audio data
import soundfile as sf
# And the openmic-vggish preprocessor
import openmic.vggish

import os

from Attention import *
from utils import cuda

def run():
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
    
    
def load(checkpoint_path):
    emb_layers = 3
    hidden_size = 128
    dropout_rate = 0.6
    model = DecisionLevelSingleAttention(
                    freq_bins=128,
                    classes_num=20,
                    emb_layers=emb_layers,
                    hidden_units=hidden_size,
                    drop_rate=dropout_rate)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model
    
    
def get_prediction(model, input):
    results = [['Instrument', 'Probability']]
    
    audio, rate = sf.read(input)

    time_points, features = openmic.vggish.waveform_to_features(audio, rate)
    X = features/255.0
    y = np.expand_dims(X, axis=0)
    X = torch.tensor(y, requires_grad=False, dtype=torch.float32)
    out = model(X)
    n_inst = 20
    instruments = ['accordion', 'banjo', 'bass', 'cello', 
                    'clarinet', 'cymbals', 'drums', 'flute', 
                    'guitar', 'mallet_percussion', 'mandolin', 'organ', 
                    'piano', 'saxophone', 'synthesizer', 'trombone', 
                    'trumpet', 'ukulele', 'violin', 'voice']
                    
    def to_numpy(x):
        return x.detach().cpu().numpy()
    
    all_predictions = torch.Tensor(0, n_inst)
    all_predictions = torch.cat((all_predictions, out.detach().cpu()))
    all_predictions = to_numpy(all_predictions)
    for i in range(n_inst):
        print('P[{:18s}] = {:.3f}'.format(instruments[i], all_predictions[0][i]))
        results.append([instruments[i], all_predictions[0][i]])
    
    return results