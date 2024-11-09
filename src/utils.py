from decimal import Decimal

def convert_floats_to_decimals(item):
    if isinstance(item, float):
        return Decimal(str(item))
    elif isinstance(item, dict):
        return {k: convert_floats_to_decimals(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_floats_to_decimals(i) for i in item]
    else:
        return item