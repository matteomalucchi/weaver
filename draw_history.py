import matplotlib.pyplot as plt


infile_dict = {
    'logs/train_puppi.log': [[],[], 'puppi'],
    #'logs/train_puppi_ef_ORedges.log': [[],[], 'puppi_ef_ORedges'],
    'logs/out_puppi_ef_oredges.txt': [[],[], 'puppi_ef_ORedges']
}


phrase = 'Train AvgLoss'

for infile, info in infile_dict.items():
    with open(infile) as f:
        f = f.readlines()

    for line in f:
        if phrase in line:
            info[0].append(float(line.split('AvgLoss: ',1)[1].split(', AvgAcc')[0]))
            info[1].append(float(line.split('AvgAcc: ',1)[1].split('\n')[0]))

    print(f'loss{info[2]}:    ', info[0])
    print(f'accuracy{info[2]}:    ', info[1])
    plt.plot(info[0], label=f'loss{info[2]}')
    plt.plot(info[1], label=f'accuracy{info[2]}')

plt.xlabel('Epoch')
plt.legend()
plt.savefig('history.png')
plt.show()
