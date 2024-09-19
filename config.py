"""
Configurations for Food Security Models
Change Variables and Hyperparams from this file
"""

#chose model to train and test, options are['bert']
MODEL_CHOICE='bert'


#set hyperparameters
MAX_LEN = 256 # 8 | 16 | 128 | 256 | 512
BATCH_SIZE = 16 # 8 | 16 | 32 | 64
NUM_FOLDS = 10
EPOCHS = 3
#CORPUS_TYPE = 'text' # text | title
LEARNING_RATE = 2e-5


#set path to relevant and irrelevant corpus folders

#rel_url='data/padi_web/relevant/relevant_articles.csv'
#irr_url='data/padi_web/irrelevant/irrelevant_articles.csv'

rel_url='data/padi_web_large/relevant/relevant_articles.csv'
irr_url='data/padi_web_large/irrelevant/irrelevant_articles.csv'
