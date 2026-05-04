from .args import ARGS
from .model import RandomModel

# Create model
model = RandomModel(ARGS.seed)


# Write results
ARGS.output.write(model.to_json())
