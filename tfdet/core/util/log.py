import numpy as np

def metric2text(info = {}, summary = None, label = None, decimal = 4):
    info = {str(k):np.round(np.mean(v, axis = tuple(np.arange(1, np.ndim(v)))) if 1 < np.ndim(v) else v, decimal) for k, v in info.items()}
    summary = ["avg"] + [np.round(np.mean(v), decimal).astype(str) for v in info.values()] if summary is None else ["summary"] + [str(np.round(v, decimal)) for v in summary]
    info = {k:v.astype(str) for k, v in info.items()}
    label = list(range(len(list(info.values())[0]))) if label is None else label
    count = [max(max(max([len(str(l)) for l in label]), len("label")), len(summary[0]))] + [max(max(len(k), max([len(_v) for _v in np.reshape(v, -1)])), len(summary[1 + i])) for i, (k, v) in enumerate(info.items())]
    line = "+" + "+".join(["-" * cnt for cnt in count]) + "+"
    header = "|" + "|".join([t + " " * (c - len(t)) for t, c in zip(["label"] + list(info.keys()), count)]) + "|"
    content = "\n".join(["|" + "|".join([t + " " * (c - len(t)) for t, c in zip([str(l)] + [np.mean(v[i]) if 0 < np.ndim(v[i]) else v[i] for v in info.values()], count)]) + "|" for i, l in enumerate(label)])
    summary = "|" + "|".join([t + " " * (c - len(t)) for t, c in zip(summary, count)]) + "|"
    form = "\n".join([line, 
                      header,
                      line,
                      content,
                      line,
                      summary,
                      line])
    return form

def concat_text(text, axis = 1, n_blank = 1):
    text = [text] if np.ndim(text) == 0 else text
    if axis == 0:
        new_text = ("\n" * (n_blank + 1)).join(text)
    else:
        text = [t.split("\n") for t in text]
        max_row = max([len(t) for t in text])
        max_cols = [max([len(t) for t in t]) for t in text]
        
        blank = " " * n_blank
        new_text = []
        for i in range(max_row):
            new_line = []
            for max_col, t in zip(max_cols, text):
                line = " " * max_col
                if i < len(t):
                    line = t[i] + line[len(t[i]):]
                new_line.append(line)
            new_text.append(blank.join(new_line))
        new_text = "\n".join(new_text)
    return new_text