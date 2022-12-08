import matplotlib.pyplot as plt
import sys

orig_stdout = sys.stdout
f = open('history.txt', 'w')
sys.stdout = f

infile_dict = {
    #'logs/train_puppi.log': [[],[], [], [], 'puppi'],
    #'logs/train_puppi_ef_ORedges.log': [[],[],[], [], 'puppi_ef_ORedges'],
    #'logs/out_puppi_ef_oredges.txt': [[],[], 'puppi_ef_ORedges']
    'logs/train_puppi_ef_ORedges_2.txt': [[],[],[], [],  'puppi_ef_ORedges_2'],
    'logs/CMSAK4_PN_20221206-135957_CMSAK4_PN_ranger_lr0.01_batch512.log': [[],[], [],[],'puppi'],
}


for infile, info in infile_dict.items():
    with open(infile) as f:
        f = f.readlines()

    for line in f:
        if 'Train AvgLoss' in line:
            info[0].append(float(line.split('AvgLoss: ',1)[1].split(', AvgAcc')[0]))
            info[1].append(float(line.split('AvgAcc: ',1)[1].split('\n')[0]))
        elif 'validation metric' in line :
            info[2].append(float(line.split('validation metric: ',1)[1].split(' (best:')[0]))
        #elif 'Test metric ' in line:
            #info[3].append(float(line.split('Test metric ',1)[1].split('\n')[0]))

    print(f'loss {info[4]}:    ', info[0])
    print(f'accuracy {info[4]}:    ', info[1])
    print(f'validation metric {info[4]}:    ', info[2])
    #print(f'Test metric {info[4]}:    ', info[3])
    plt.plot(info[0], label=f'loss {info[4]}')
    plt.plot(info[1], label=f'accuracy {info[4]}')
    plt.plot(info[2], label=f'validation metric{ info[4]}')
    #plt.plot(info[3], label=f'Test metric {info[4]}')

plt.xlabel('Epoch')
plt.legend()
plt.savefig('history.png')
plt.show()
