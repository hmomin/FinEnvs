# Data Format

Individual financial instruments are separated into training, cross-validation, and testing data files. Training files may contain millions of lines spanning decades-worth of data.

Each file is expected to be a comma-delimited csv file.

Each line is ordered by the following fields:

_Date, Time, Open, High, Low, Close, Volume_

Time values are in the US Eastern Time (ET) time zone. Time values indicate when that bar opened. For example, a time-stamp of 10:34 AM is for the period between 10:34 AM and 10:35 AM. Pre-market (8:00-9:30 AM) and after-market (4:00-6:30 PM) sessions may be included as well.

All data is adjusted for stock splits and dividends.

If there were no transactions during a specific time interval, the data is not recorded for that interval.
