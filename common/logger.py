def safe_print(*args, **kwargs):
    cleaned = []
    for a in args:
        if isinstance(a, str):
            cleaned.append(a.encode("utf-8", errors="replace").decode("utf-8"))
        else:
            cleaned.append(a)
    print(*cleaned, **kwargs)