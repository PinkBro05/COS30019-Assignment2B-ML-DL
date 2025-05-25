import matplotlib.pyplot as plt
import csv
import os

data = os.path.join(os.path.dirname(__file__),'gru loss.csv')

reader = csv.reader(open(data, 'r'))
loss = []
val_loss = []
for row in reader:
    if reader.line_num == 1:
        continue
    else:
        loss.append(float(row[0]))
        val_loss.append(float(row[2]))

loss_data = [loss, val_loss]

fig = plt.figure()
ax = fig.add_subplot(111)

for i, data in enumerate(loss_data):
    label = 'Loss' if i == 0 else 'Validation Loss'
    ax.plot(data, marker='o', linestyle='-', label=label)
plt.plot(loss, marker='o', linestyle='-', color='b')
plt.title('GRU Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
