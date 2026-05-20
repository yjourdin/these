from ..utils import file_or_stdout
from .args import ARGS
from .model import RandomModel

# Create model
model = RandomModel(ARGS.seed)


# Write output
with file_or_stdout(ARGS.output, "w") as f:
    f.write(model.to_json())
