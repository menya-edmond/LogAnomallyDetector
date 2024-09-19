import sys
import torch
import numpy as np
from optparse import OptionParser
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


if __name__ == "__main__":
    from data import dataGenerator
    import config as cfg
    from models import modelTrainTest as mtt
    from data.profile import Profile
    
    if not sys.argv:
        modelChoice=sys.argv[0]
        
    else:
        optparser = OptionParser()
        optparser.add_option('-a', '--modelChoice',
                             dest='modelChoice',
                             help='select model',
                             default=cfg.MODEL_CHOICE,
                             type='string')
        (options, args) = optparser.parse_args()
    
    modelChoice=options.modelChoice
    rel_data=cfg.rel_url
    irr_data=cfg.irr_url
    max_len=cfg.MAX_LEN
    batch_size=cfg.BATCH_SIZE
    num_folds=cfg.NUM_FOLDS
    epochs=cfg.EPOCHS
    #corpus_type=cfg.CORPUS_TYPE

    if modelChoice == 'bert':
        FOUNDATIONAL_LLM = "google-bert/bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(FOUNDATIONAL_LLM)



    # tell Pytorch to use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    if torch.cuda.device_count():
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    print('We are running ', modelChoice)
    
    #data = dataGenerator.data_generator(rel_data,irr_data,corpus_type)
    predictions = ["node had unexpected reboot at this time according to dcm history",
                   "[Verbose] ContainerId : 99b50dc8-aa6c-4f4d-81d1-fb95041b7801, TenantName : bc24f0e4-0640-48f2-bc11-94f7da9f430e, RCA: CustomerInitiated.ContainerOperation.ContainerCreated CreateTenant,True, (RoleType: IaaSDurable)"] #current timestep bad log

    references = ["node made pxe request at this time according to dcm history",
                  "basicutils.py - set_environment_variable() - env:POLICY(after set value) = nocache_noetl_dyn_mm_CloudHost_Ni_Amd64"] #all logs within 2 hours timeframe

    similarity_scores = mtt.learn_similarity(epochs,device,modelChoice,predictions,references)#call the model and compute similarities

    print(similarity_scores)



