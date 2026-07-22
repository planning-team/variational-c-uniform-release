def seconds_to_human_readable_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = round(seconds % 60)

    return f"{hours:02}h {minutes:02}m {remaining_seconds:02}s"


def calculate_zeros_pad(data_length: list) -> int:
    return len(str(data_length)) + 1


def zfill_zeros_pad(idx: int, n_zeros: int) -> str:
    return str(idx).zfill(n_zeros)
