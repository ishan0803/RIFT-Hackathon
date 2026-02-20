import pandas as pd
df = pd.DataFrame({"timestamp": [pd.Timestamp("2021-01-01"), None, pd.Timestamp("2021-01-02")]})
df = df.assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], utc=True, errors="coerce"))
print(df["timestamp"].dtype)
diffs = df["timestamp"].diff(periods=1)
print(diffs)
print(diffs <= pd.Timedelta(hours=72))
