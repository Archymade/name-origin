from torch.nn.utils.rnn import PackedSequence
from torch import Tensor

from torch import save
from torch import load

from typing import Callable, Tuple, Optional

def make_predictions(string: Optional[Tuple[str, list]], model: Callable[[Tensor, PackedSequence], Tensor]) -> dict:
    '''
    Make predictions using trained model.
    
    Parameters
    ----------
    string
        Model input.
    model
        Trained model.
        
    Returns
    -------
    ret
        Dictionary of prediction(s).
    '''
    if type(string) == str or len(string) == 1:
        string_tensor = data.string2tensor(string)
        X_ = string_tensor.unsqueeze(0)
        pred = torch.exp(model(X_))
        pred = pred.max(dim = -1).indices.item()
        pred = dict(string = data.int2label[pred])
    else:
        string_tensor = [data.string2tensor(s) for s in string]
        lengths = [len(s) for s in string]
        X_ = torch.nn.utils.rnn.pad_sequence(string_tensor, batch_first = True)
        X_ = torch.nn.utils.rnn.pack_padded_sequence(X_, lengths = lengths, enforce_sorted = False, batch_first = True)
        pred = torch.exp(model(X_))
        pred = pred.max(dim = -1).indices
        pred = [data.int2label[p.item()] for p in pred]
        pred = dict(zip(string, pred))
    
    
    return pred

def save_checkpoint(path_to_file, epochs, model, optimizer, history, scheduler = None):
    '''
    Save artefact checkpoints.
    '''
    checkpoint = {
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'epochs' : epochs,
                    'history' : history
                }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    with open(path_to_file, 'wb') as file:
        save(checkpoint, file)
    return None

def load_checkpoint(path: str, artefacts, keys: Optional[Tuple[list, str]]):
    '''
    Load saved checkpoints from disk into artefacts.
    '''
    assert len(artefacts) == len(keys), 'Number of artefacts must equal number of keys.'
    ckpt = load(path)
    
    for artefact, key in zip(artefacts, keys):
        artefact.load_state_dict(ckpt[key])
    return artefacts