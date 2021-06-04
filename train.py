import torch
import torch.nn as nn
import model_l1_l1
import data_loader as dl
import matplotlib.pyplot as plt
import plot_utils
import sys

args = sys.argv
print("compression rate: "+ str((1/float(args[1])) * 100)+"%, divider: " + str(args[2])+" , K: " + str(args[3])) 
input_size = 256
if input_size == 256:
	path = 'movingMnist/mnist_test_seq_16.npy'
else:
	path = '/movingMnist/mnist_test_seq.npy'

time_steps = 20
data_loader = dl.Moving_MNIST_Loader(path=path, time_steps=time_steps, flatten=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

model = model_l1_l1.l1_l1(input_size, float(1/float(args[1])), int(args[2]), int(args[3]), 4*input_size, batch_size).to(device)
learning_rate = 0.001
epochs = 60
training_samples = 8000
test_samples = 1000
optimizer = torch.optim.Adam(model.parameters, lr=learning_rate)#TODO change with all trainable params

def fit(model, dataloader):
	model.train()
	loss = 0.0
	iterations = int(training_samples/batch_size)
	training_loss = []
	for epoch in range(epochs):
		loss = 0.0
		print("epoch: " + str(epoch))
		for j in range (iterations):
			#print(j)
			data = dataloader.get_batch("train", batch_size)
			input_ = torch.tensor(data, dtype=torch.float32, device=device)
			optimizer.zero_grad()
			output = model.forward(input_)
			if j==0 and epoch==0:
				print(compute_psnr(input_[0], output[0]))
			loss_func = torch.mean((output-input_) ** 2)
			loss += loss_func.item()
			#print(loss_func.item())
			loss_func.backward()
			optimizer.step()
			if j == (iterations-1) and (epoch%50 == 0 or epoch == epochs-1):
				# !!!!! comment out this lines if you dont want to save frames
				filename = "endofepoch" + str(epoch) +".png"
				#print(epoch)
				plot_utils.save_frame(output, filename)
                #TODO add some evaluation
				
		#print("epoch loss: " + str(loss))
		#print(loss)
		
		training_loss.append(loss)
		#if training_loss[len(training_loss)-2]<loss:
			#print("validation psnr:")
			#print(validation(model, dataloader))
				#if validation(model, dataloader)>=30:
				#plot_utils.save_frame(output, "early_stopping.png")
				#break
	return training_loss

def validation(model, dataloader):
	val_psnr = []
	model.eval()
	with torch.no_grad():
		iterations = int(test_samples/batch_size)
		#print(iterations)
		for i in range (iterations): 
			loss = 0.0
			validation_data = dataloader.get_batch("validation", batch_size)
			input_ = torch.tensor(validation_data,  device=device)
			output = model.forward(input_)
			loss = torch.mean((output - input_) ** 2).item()
			for frame in range(time_steps):
				for j in range (batch_size):
					val_psnr.append(compute_psnr(input_[frame][j], output[frame][j]))			
			#print ("validation loss: " + str(loss))
	return (sum(val_psnr)/len(val_psnr)).item()

def test(model, dataloader):
	testing_psnr = []
	model.eval()
	with torch.no_grad():
		iterations = int(test_samples/batch_size) + 1
		#print(iterations)
		for i in range (iterations): 
			loss = 0.0
			test_data = dataloader.get_batch("test", batch_size)
			input_ = torch.tensor(test_data,  device=device)
			output = model.forward(input_)
			loss = torch.mean((output - input_) ** 2).item()
			for frame in range(time_steps):
				for j in range (batch_size):
					testing_psnr.append(compute_psnr(input_[frame][j], output[frame][j]))
			# !!!!! comment out this lines if you dont want to save frames
			#plot_utils.save_sequences(input_, 0, output, "test_sequence0" + str(i) +".png")
			plot_utils.save_sequences(input_, 1, output, "test_sequence1" + str(i) +".png")
			#plot_utils.save_sequences(input_, batch_size -1 , output, "test_sequence64" + str(i) +".png")
			
			#print ("test loss: " + str(loss))
	return testing_psnr

def compute_psnr(frame1, frame2):
	mse = torch.mean((frame1 - frame2) ** 2)
	#print(mse)
	psnr = 20 * torch.log10(255/torch.sqrt(mse))
	return psnr

def anomaly_classifier(model, dataloader, threshold):
	model.eval()
	with torch.no_grad():
		iterations = int(test_samples/batch_size)
		#print(iterations)
		labels = []
		predictions = []

		for i in range (iterations): 
			loss = 0.0
			test_data, batch_labels = dataloader.get_batch("anomaly", batch_size)
			input_ = torch.tensor(test_data,  device=device)
			output = model.forward(input_)
			for j in range (batch_size):
				labels.append(batch_labels[j])
				psnr = 0
				for frame in range(time_steps):
					tmp = compute_psnr(input_[frame][j], output[frame][j])
					psnr += tmp.item() 
				avg_video_psnr = psnr/time_steps
				if avg_video_psnr < threshold:
					print(avg_video_psnr)
					predictions.append(1)
				else:
					predictions.append(0)
		evalueate_detector(labels, predictions)


def evalueate_detector(labels, predictions):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for i in range(len(labels)):
		if labels[i] == 1 and predictions[i] == 1:
			tp += 1
		elif labels[i] == 0 and predictions[i] == 0:
			tn += 1
		elif labels[i] == 0 and predictions[i] == 1:
			fp += 1
		elif labels[i] == 1 and predictions[i] == 0:
			fn += 1
	print("TP " + str(tp) + " TN " + str(tn) + " FP " + str(fp) + " FN " + str(fn))

#print(model.Dict_D)
#torch.autograd.set_detect_anomaly(True)
training_loss = fit(model,data_loader)
#print(training_loss)
plt.figure(1)
plt.plot(training_loss)
plt.ylabel("epoch loss")
plt.xlabel("epoch")
plt.draw()
# !!!!! comment out this lines if you dont want to save the loss plot
compression_rate = int((model.matrix_A.shape[0])/(model.matrix_A.shape[1])*100)
plt.savefig("training_lossCompressionRate"+str(compression_rate)+"_Dict256*"+str(model.hidden_size)+"K" + str(model.hidden_layers) +".png")
plt.close()

psnr = test(model, data_loader)
avg_psnr = sum(psnr)/len(psnr) 
print("average psnr on test frames: " +str(avg_psnr.item()) + " dB")
anomaly_classifier(model, data_loader, (avg_psnr.item() - 5))
