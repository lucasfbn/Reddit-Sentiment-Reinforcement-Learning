def update_check_key(input_dict, update_dict):
    for key in update_dict.keys():
        if key not in input_dict.keys():
            raise ValueError(f"Invalid key: {key}")
    input_dict.update(update_dict)
    return input_dict


def dt_to_timestamp(time):
    if time is None: return None
    return int(time.timestamp())
