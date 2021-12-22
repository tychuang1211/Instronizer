import soundfile as sf
import openmic.vggish
import torch
import numpy as np

from classifier.models.Attention import DecisionLevelSingleAttention

def cuda(X):
    if type(X) is list:
        X = X[0].cuda(), X[1].cuda()
    else: 
        X = X.cuda()
    return X

def to_numpy(x):
    return x.detach().cpu().numpy()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path)

    # Load model state
    model.load_state_dict(state_dict)
    print('[attentionMic.py] using checkpoint: \n{}'.format(checkpoint_path))
    model.to(device)

    # Switch model to evaluate mode (validatin/testing)
    # Extremely important!
    model.eval()

    return model
    
def get_prediction(model, audio_path, start, end):
    info = sf.info(audio_path)
    sr = info.samplerate
    frames = info.frames
    duration = info.duration
        
    start = int(start * sr)
    end = int(end * sr)
    # Load audio
    audio, rate = sf.read(audio_path, start=start, stop=end)
    
    # Extract feature
    time_points, features = openmic.vggish.waveform_to_features(audio, rate)
    features = np.expand_dims(features, axis=0)
    X = features/255.0
    X = torch.tensor(X, requires_grad=False, dtype=torch.float32)
    if torch.cuda.is_available():
        X = cuda(X)
    
    outputs = model(X)
    
    instruments = ['accordion', 'banjo', 'bass', 'cello', 'clarinet', 'cymbals',
                    'drums', 'flute', 'guitar', 'mallet_percussion', 'mandolin',
                    'organ', 'piano', 'saxophone', 'synthesizer', 'trombone',
                    'trumpet', 'ukulele', 'violin', 'voice']
    
    for i in range(20):
        print('P[{:18s}=1] = {:.3f}'.format(instruments[i], outputs[0][i]))
    results = to_numpy(outputs[0])
    print('Dominant instrument: ', instruments[results.argmax()])
    return results.tolist()