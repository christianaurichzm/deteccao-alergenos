import cachetools

cache = cachetools.LFUCache(maxsize=1000)


def get_cached_transformation(sentence, transformation_func):
    hash_key = hash(sentence)

    if hash_key in cache:
        return cache[hash_key]

    transformed = transformation_func(sentence)

    cache[hash_key] = transformed

    return transformed
