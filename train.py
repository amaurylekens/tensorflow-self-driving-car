import numpy as np 
from PIL import Image
import csv
from SDC_CNN import SDC_CNN
import scipy

# Load the data
data = []
true_values = []
with open('../training_3/driving_log.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
	# Extract the images from the paths given in the csv file en resize them to keep the minimum useful information
    for row in reader:
	    image = np.asarray(Image.open(row[0]))
	    image_resized = scipy.misc.imresize(image, [66, 200])
	    data.append(image_resized)
		# Retrieve the correct value of the angle of the steering wheel for this image
	    true_values.append(row[3])

data = np.asarray(data)
true_values = np.asarray(true_values)
print("data loaded")

# select data to balance zero and non-zero. Keep only 1000 inputs where the value of the steering wheel is equal to 0
index_of_non_0 = np.where(true_values != '0')[0]
index_of_0 = np.where(true_values == '0')[0][0:1000]
index = np.array(list(index_of_0) + list(index_of_non_0))
np.random.shuffle(index)

true_values = true_values[index]
data = data[index]

print(len(true_values))
input_dim = list(data[1,:,:,:].shape)

# Create the convolutional neural network
sdc_cnn = SDC_CNN(input_dim)
sdc_cnn.train(data, true_values)





