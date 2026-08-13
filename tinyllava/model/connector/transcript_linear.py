import torch.nn as nn

from . import register_connector
from .base import Connector


@register_connector('transcript_linear')
class TranscriptLinearConnector(Connector):
    """Linear projection from a transcriptomics-encoder embedding
    (`config.vision_hidden_size`, 515 for the locked BulkFormer-93M) to the LLM
    embedding dim (`config.hidden_size`). Functionally identical to the stock
    `linear` connector; named separately as the transcriptomic pipeline's
    extension point. See integration/repo_findings.md §6.
    """

    def __init__(self, config):
        super().__init__()
        self._connector = nn.Linear(config.vision_hidden_size, config.hidden_size)
