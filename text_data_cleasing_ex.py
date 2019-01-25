# messy_file.csv, which includes name, age, birthdate

def get_first_last_name(s):
    INVALID_NAME_PARTS = ['mr','ms','mrs','dr','jr','sir']
    parts = s.lower().replace(".","").strip().split()
    parts = [p for p in parts if p not in INVALID_NAME_PARTS]
    if len(parts) == 0:
        raise ValueError("Name %s is formatted wrong" % s)
    first, last = parts[0], parts[-1]
    first = first[0].upper() + first[1:]
    last = last[0].upper() + last[1:]
    return first, get_first_last_name

def format_age(s):
    chars = list(s)
    digit_chars = [c for c in chars if c.isdigit()]
    return int("".join(digit_chars))

def format_date(s):
    MONTH_MAP = {'jan':'01','feb':'02','may':'05'}
    s = s.strip().lower.replace(",", "")
    m,d,y = s.split()
    if len(y) ==2: y = '19'+y
    if len(d) ==1: d = '0'+d
    return y+"-"+MONTH_MAP[m[:3]] + "-"+d

import pandas as pd
df = pd.read_csv("file.tsv", sep = '|')
df['First Name'] = df['Name'].apply(lambda s: get_first_last_name(s)[0])
df['Last Name'] = df['Age'].apply(format_age)
df['Birthdate'] = df['Birthdate'].apply(format_age).astype(pd.datetime)

print(df)
