from .argument_parser import ScriptEnum, parse_args
from .function import generate_linext, generate_weak_order, generate_weak_order_ext

SCRIPT_DICT = dict(
    zip(ScriptEnum, (generate_linext, generate_weak_order, generate_weak_order_ext))
)

# Parse arguments
args = vars(parse_args())


# Run script
script = args.pop("script")
print(SCRIPT_DICT[script](**args), end="")
