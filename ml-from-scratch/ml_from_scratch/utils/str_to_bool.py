def parse_str_arg_to_bool(dict_args: dict, param: str) -> bool:
    try:
        param_value = dict_args.get(f"{param}")
        return param_value.lower().strip() in ['true', '1', 't', 'y', 'yes']
    except KeyError as k:
        raise KeyError(f"'{param}' arg not found in arguments namespace. {k}")