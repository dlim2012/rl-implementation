
def time_hr_min_sec(t):
    hr = t // 3600
    t -= hr * 3600
    min = t // 60
    t -= min * 60
    sec = int(t)
    return (hr, min, sec)