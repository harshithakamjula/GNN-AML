from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
import sys
import os

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = "/content/drive/MyDrive/IBM/LI-Small_Trans.csv"
outPath = "/content/drive/MyDrive/IBM/Formattedtransactions/Formatted_LI-Small_Trans.csv"

raw = dt.fread(inPath, columns = dt.str32)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,Is Laundering\n"

firstTs = -1

with open(outPath, 'w') as writer:
    writer.write(header)
    for i in range(raw.nrows):
        datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
        cur2 = get_dict_val(raw[i,"Payment Currency"], currency)


        fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

        fromAccIdStr = raw[i,"From Bank"] + raw[i,2]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw[i,"To Bank"] + raw[i,4]
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedUSD = raw[i,"Amount Received"]
        amountPaidUSD = raw[i,"Amount Paid"]

        amountReceivedOrig = float(amountReceivedUSD) if isinstance(amountReceivedUSD, str) and amountReceivedUSD.replace(".", "").isdigit() else 0.0
        amountPaidOrig = float(amountPaidUSD) if isinstance(amountPaidUSD, str) and amountPaidUSD.replace(".", "").isdigit() else 0.0


        isl = int(raw[i,"Is Laundering"])

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

        writer.write(line)

formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]
formatted.to_csv(outPath)
