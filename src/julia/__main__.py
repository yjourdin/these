from .argument_parser import ScriptEnum, parse_args

# Parse arguments
args = vars(parse_args())


# Run script
script = args.pop("script")
print(ScriptEnum[script](**args))
