""" 
trainer/builder.py file adapted from Tencent Youtu
[TencentYoutuResearch/Classification-SemiCLS](https://github.com/TencentYoutuResearch/Classification-SemiCLS) repository.
This file build the meta archs for the framework
Each training must register here

The modifications introduced in this file register a new trainer CoMatchTriplet.
"""

from copy import deepcopy
from functools import partial

from .classifier import Classifier
from .comatch import CoMatch 
from .comatch_triplet import CoMatchTriplet # added for CoMatch+Triplet
from .fixmatch import FixMatch
from .fixmatch_ccssl import FixMatchCCSSL
from .comatch_ccssl import CoMatchCCSSL

# meta archs for all trainers
meta_archs = {
    "FixMatch": FixMatch,
    "CoMatch": CoMatch,
    "CoMatchTriplet": CoMatchTriplet, # added for CoMatch+Triplet
    "Classifier": Classifier,
    "FixMatchCCSSL": FixMatchCCSSL,
    "CoMatchCCSSL":CoMatchCCSSL,
}


def build(cfg):
    """ build function for trainer
        the cfg must contain type for trainer
        other configs will be used as parameters
    """
    trainer_cfg = deepcopy(cfg)
    type_name = trainer_cfg.pop("type")
    return partial(meta_archs[type_name], cfg=trainer_cfg)
