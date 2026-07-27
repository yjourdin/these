from functools import reduce

from src.random import rng_

from ..utils import add_filename_suffix, file_or_stdout
from .args import ARGS
from .model import SRMPModel, SRMPParamFlag, srmp_group_model

models = []

# Create model
if ARGS.group_size == 1:
    model = SRMPModel.random(
        nb_profiles=ARGS.k,
        nb_crit=ARGS.m,
        rng=ARGS.seed,
        profiles_values=ARGS.profiles_values,
    )
else:
    if any((ARGS.reference, ARGS.profile_amp, ARGS.weight_amp, ARGS.lex_amp)):
        if ARGS.reference:
            with ARGS.reference.open("r") as f:
                model = SRMPModel.from_json(f.read())
        else:
            model = SRMPModel.random(
                nb_profiles=ARGS.k,
                nb_crit=ARGS.m,
                rng=ARGS.seed,
                profiles_values=ARGS.profiles_values,
            )

        models = [
            SRMPModel.from_reference(
                model,
                ARGS.profile_amp or 0,
                ARGS.weight_amp or 0,
                ARGS.lex_amp or 0,
                rng=rng,
            )
            for rng in rng_(ARGS.seed).spawn(ARGS.group_size)
        ]
    else:
        model_class = srmp_group_model(
            reduce(lambda x, y: x | y, ARGS.shared, SRMPParamFlag.NONE)
        )
        model = model_class.random(
            group_size=ARGS.group_size,
            nb_profiles=ARGS.k,
            nb_crit=ARGS.m,
            rng=ARGS.seed,
            profiles_values=ARGS.profiles_values,
        )


# Write output
if models:
    for model, output in zip(
        models,
        (
            ARGS.output and add_filename_suffix(ARGS.output, f"_{i}")
            for i in range(ARGS.group_size)
        ),
    ):
        with file_or_stdout(ARGS.output, "w") as f:
            f.write(model.to_json())
else:
    with file_or_stdout(ARGS.output, "w") as f:
        f.write(model.to_json())
