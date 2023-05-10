import sys
sys.path.append(r"C:\Users\TDT\Downloads\photometry-scripts")
sys.path.append(r"C:\Users\TDT\Downloads\photometry-scripts\extras")
from import_lick_phot import *


#root folder location
lick_data_f = Path(r'C:\Users\TDT\Documents\read_lickometers\lickometer_data')


#folders for each trainingday 
train_d1 = lick_data_f/'2022'/'202203'/'20220303'
train_d2 = lick_data_f/'2022'/'202203'/'20220304'
train_d3 = lick_data_f/'2022'/'202203'/'20220305'


#folders for each testing day 
test_d1 = lick_data_f/'2022'/'202203'/'20220307'
test_d2 = lick_data_f/'2022'/'202203'/'20220308'


# BEYOND HERE DONT TOUCH!
training = [ train_d1, train_d2, train_d3 ]
testing = [ test_d1, test_d2]


def main():

    training_data = []
    testing_data = []
    for i,v in enumerate(training):
        for j in v.iterdir():
            mice = j.name.split('_')[1:]
            sess = 'am' if 'am' in j.name.lower() else 'pm'
            d = open_save_lick(j, mouse1 = mice[0],
                               mouse2 = mice[1],
                               cond1 = f'training_{i+1}_{sess}',
                               cond2 = f'training_{i+1}_{sess}')
            training_data.extend(d)
    for i,v in enumerate(testing):
        for j in v.iterdir():
            mice = j.name.split('_')[1:]
            d = open_save_lick(j, mouse1 = mice[0],
                               mouse2 = mice[1],
                               cond1 = f'testing_{i+1}',
                               cond2 = f'testing_{i+1}')
            testing_data.extend(d)
    np.save('training.npy',training_data)
    np.save('testing.npy',testing_data)
             

if __name__ == '__main__':
	main()
