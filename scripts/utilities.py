from torch import save
from torch import load



def save_checkpoint(path_to_file, epochs, model, optimizer, history, scheduler = None):
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

def load_checkpoint(path, artefacts, keys):
    '''
    Load saved checkpoints from disk into artefacts.
    '''
    assert len(artefacts) == len(keys), 'Number of artefacts must equal number of keys.'
    ckpt = load(path)
    
    for artefact, key in zip(artefacts, keys):
        artefact.load_state_dict(ckpt[key])
    return artefacts