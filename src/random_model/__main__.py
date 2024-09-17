from .argument_parser import parse_args
from .model import RandomModel

# Parse arguments
args = parse_args()


# Create model
model = RandomModel(args.seed)


# Write results
args.output.write(model.to_json())
