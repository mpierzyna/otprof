"""SQF-H: the height-aware SqueezeFormer of the paper (`data`: what it makes of the dataset files, `model`: the
network, `lit`: its training).

The published SQF-H (decoupled, trained on the nl domain, with its output quantile mapping) is a Hugging Face model
repo of its own; other models get their own repos:

    from otprof.models import sqf_h
    from otprof.predict import load_model

    predictor = load_model(sqf_h.HF_REPO_ID, revision=sqf_h.HF_REVISION)
"""

HF_REPO_ID = "maxiimilian/otprof-sqfh"
# Commit of the repo this otprof version uses, so a model update never changes the results of an installed version.
# None (latest) until the first upload.
HF_REVISION: str | None = None
