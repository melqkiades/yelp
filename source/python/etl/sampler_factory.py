from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks


def create_sampler(sampler_name, random_state=None):

    if sampler_name is None:
        return None
    if sampler_name.lower() == 'random_under_sampler':
        return RandomUnderSampler(random_state=random_state)
    if sampler_name.lower() == 'tomek_links':
        TomekLinks(random_state=random_state),
    if sampler_name.lower() == 'enn':
        EditedNearestNeighbours(random_state=random_state)
    if sampler_name.lower() == 'ncl':
        NeighbourhoodCleaningRule(random_state=random_state)
    if sampler_name.lower() == 'random_over_sampler':
        return RandomOverSampler(random_state=random_state)
    if sampler_name.lower() == 'smote':
        SMOTE(random_state=random_state)
    if sampler_name.lower() == 'smote_tomek':
        SMOTETomek(random_state=random_state)
    if sampler_name.lower() == 'smote_enn':
        SMOTEENN(random_state=random_state)
    else:
        raise ValueError('Unsupported value \'%s\' for sampler' % sampler_name)
