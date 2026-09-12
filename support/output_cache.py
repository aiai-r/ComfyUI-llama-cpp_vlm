import hashlib
import json


def _stable_plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _stable_plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_stable_plain(item) for item in value]
    return repr(value)


def _image_bytes_payload(image):
    value = image
    for method_name in ("detach", "cpu", "contiguous"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()

    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()

    tobytes_method = getattr(value, "tobytes", None)
    if callable(tobytes_method):
        return {
            "shape": repr(getattr(value, "shape", None)),
            "dtype": repr(getattr(value, "dtype", None)),
            "bytes": tobytes_method(),
        }

    if isinstance(value, bytes):
        return {"shape": None, "dtype": "bytes", "bytes": value}

    encoded = json.dumps(_stable_plain(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"shape": None, "dtype": type(value).__name__, "bytes": encoded}


def fingerprint_images(images):
    if images is None:
        return None

    digest = hashlib.sha256()
    image_items = images if isinstance(images, (list, tuple)) else (images,)
    digest.update(str(len(image_items)).encode("utf-8"))
    for image in image_items:
        payload = _image_bytes_payload(image)
        digest.update(str(payload["shape"]).encode("utf-8"))
        digest.update(str(payload["dtype"]).encode("utf-8"))
        digest.update(payload["bytes"])
    return digest.hexdigest()


def make_instruct_cache_key(
    *,
    llama_model,
    preset_prompt,
    custom_prompt,
    system_prompt,
    inference_mode,
    max_frames,
    max_size,
    seed,
    parameters,
    output_think_block,
    images,
    cache_image_analysis=False,
):
    payload = {
        "llama_model": _stable_plain(llama_model),
        "preset_prompt": preset_prompt,
        "custom_prompt": custom_prompt,
        "system_prompt": system_prompt,
        "inference_mode": inference_mode,
        "max_frames": max_frames,
        "max_size": max_size,
        "seed": seed,
        "parameters": _stable_plain(parameters),
        "output_think_block": bool(output_think_block),
        "images": fingerprint_images(images),
        "cache_image_analysis": cache_image_analysis,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_cached_instruct_output(cache, uid, cache_key):
    entry = cache.get(str(uid))
    if not entry or entry.get("key") != cache_key:
        return None

    out1, out2, state_uid = entry["output"]
    return (out1, list(out2), state_uid)


def store_instruct_output(cache, uid, cache_key, output):
    out1, out2, state_uid = output
    cache[str(uid)] = {
        "key": cache_key,
        "output": (out1, list(out2), state_uid),
    }
