from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks


def create_sampler(sampler_name, random_state=None):

    if sampler_name is None or sampler_name == 'None':
        return None
    if sampler_name.lower() == 'randomundersampler':
        return RandomUnderSampler(random_state=random_state)
    if sampler_name.lower() == 'tomeklinks':
        return TomekLinks(random_state=random_state),
    if sampler_name.lower() == 'enn':
        return EditedNearestNeighbours(random_state=random_state)
    if sampler_name.lower() == 'ncl':
        return NeighbourhoodCleaningRule(random_state=random_state)
    if sampler_name.lower() == 'randomoversampler':
        return RandomOverSampler(random_state=random_state)
    if sampler_name.lower() == 'smote':
        return SMOTE(random_state=random_state)
    if sampler_name.lower() == 'smotetomek':
        return SMOTETomek(random_state=random_state)
    if sampler_name.lower() == 'smoteenn':
        return SMOTEENN(random_state=random_state)
    else:
        raise ValueError('Unsupported value \'%s\' for sampler' % sampler_name)
