def chunk_list(lst, size):
    """Split a list into chunks of the given size."""
    chunks = []
    for i in range(0, len(lst), size + 1):
        chunks.append(lst[i:i + size])
    return chunks


def flatten(nested):
    """Flatten a list of lists into a single list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
