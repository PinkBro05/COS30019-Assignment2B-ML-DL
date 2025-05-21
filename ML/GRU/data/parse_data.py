import csv
import os

fields = ['Time', 'Flow', 'Points', 'Observed']

data = os.path.expanduser('~/AI/COS30019-Assignment2B-ML-DL/Data/Raw/main/Scat_Data.csv')

output = os.path.join(os.path.dirname(__file__),'TrainingDataAdaptedOutput.csv')

train_dir = os.path.join(os.path.dirname(__file__),'train.csv')
test_dir = os.path.join(os.path.dirname(__file__),'test.csv')

reader = csv.reader(open(data, 'r'))
train_writer = csv.writer(open(output, 'w', newline=''))
# row_count = sum(1 for row in reader)
times = []

# lines = reader
for row in reader:
    if reader.line_num == 1:
        times = list(row[10:-3])
    elif reader.line_num == 2:
        continue
    else:
        values = []
        date = row[9]
        location = row[0]
        num_col = len(row)
        for col_index in range(len(times)):
            values.append(date + " " + times[col_index])
            values.append(row[col_index + 10])
            values.append(1)
            values.append(100)
            values.append(location)
            
            train_writer.writerow(values)
            
            values = []

fid = open(output, "r")
li = fid.readlines()

test_data_size = 72000
row_count = len(li)

header = "15 Minutes,Lane 1 Flow (Veh/15 Minutes),# Lane Points,% Observed,SCATS\n"

train = li[:(row_count - test_data_size)]
train.insert(0,header)
test = li[(row_count - test_data_size):]
test.insert(0,header)

fid.close()

train_file = open(train_dir, "w")
train_file.writelines(train)
train_file.close()

test_file = open(test_dir, "w")
test_file.writelines(test)
test_file.close()
