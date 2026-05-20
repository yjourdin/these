from .args import parse_args
from .script_enum import ScriptEnum

# Parse arguments
args = vars(parse_args())


# Run script
script = args.pop("script")
print(ScriptEnum[script](**args))
