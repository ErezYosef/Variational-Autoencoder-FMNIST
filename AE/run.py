from modules import *
#from modules import class_counter_ds
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim as optim
cuda_is_avail = torch.cuda.is_available()

from torchvision.datasets import MNIST, FashionMNIST

flatten = lambda x: transforms.ToTensor()(x).view(-1)


train_set = FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = flatten)

test_set = FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = flatten)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, num_workers=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, num_workers=2, shuffle=True)

# ===== To create and train models:

# Run:
cuda_is_avail = torch.cuda.is_available()
epochs = 25
lr = 1e-3
wd = 1e-1
text = 'regular'
fkl = 1
tag = f'-{lr}-fkl{fkl}'

model = Autoencoder(x_dim=784, z_dim=50, h_dim=[600, 600])

time_stamp = datetime.now().strftime("%b%d,%H:%M")
log_dir_name = 'AE/runs/' + time_stamp + tag
tb = SummaryWriter(log_dir=log_dir_name)
with open(f"{log_dir_name}/Output.txt", "a") as text_file:
    text_file.write(f' ---- Autoencoder(x_dim=784, z_dim=50, h_dim=[600, 600])  :\n')
    text_file.write(f' ---- epochs:{epochs},\tlr: {lr},\twd: {wd}  :\n')
    text_file.write(f' ---- {text}  :\n')

begin_run_time = time.time()
'''
def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)
'''
if cuda_is_avail:
    model.cuda()
    print('model on cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
binary_CE = torch.nn.BCELoss(reduction='sum')  # BCELoss

m = len(train_set)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for (images, _) in train_loader:
        # images = Variable(images)

        if cuda_is_avail:
            images = images.cuda()

        reconstruction = model(images)

        bce_loss = binary_CE(reconstruction, images)

        L = bce_loss + model.kl_divergence * fkl
        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        total_loss += L.item()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for tstdata, _ in test_loader:

            if cuda_is_avail:
                tstdata = tstdata.cuda()
            reconstruction = model(tstdata)

            # sum up batch loss
            bce_loss = binary_CE(reconstruction, tstdata)
            L = bce_loss + model.kl_divergence * fkl
            test_loss += L.item()
            # test_loss += loss_function(recon, data, mu, log_var).item()

    test_loss = test_loss / len(test_loader.dataset)

    total_loss = total_loss / len(train_loader.dataset)

    tb.add_scalar('loss/train', total_loss, epoch)
    tb.add_scalar('loss/test', test_loss, epoch)
    with open(f"{log_dir_name}/Output.txt", "a") as text_file:
        text_file.write(f'Epoch:{epoch},\tLoss: {total_loss:.2f},\ttstLoss: {test_loss:.2f}\n')

    print(f"Epoch: {epoch}\tLoss: {total_loss:.2f},\ttstLoss: {test_loss:.2f}")

tb.close()
end_run_time = time.time() - begin_run_time
print(f"time: {end_run_time:.2f}")
print("saved to dir: ", log_dir_name)
with open(f"{log_dir_name}/Output.txt", "a") as text_file:
    text_file.write(f'Total Time: {end_run_time:.2f}\n')

torch.save(model.state_dict(), f'{log_dir_name}/model_params.pt')
print ("saved to dir: ", log_dir_name)

##### RUN BEFORE SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
from plotcm import *



tst_loader = torch.utils.data.DataLoader(test_set, batch_size=10000)
(tst_data,tst_label) = next(iter(tst_loader))
tst_label = tst_label.numpy().reshape(-1)

savefig = False
kernel = 'rbf'
size_data = [100, 600, 1000, 3000]
seed = -1
for size in size_data:
    labeles_per_class = size / 10

    subdata = sampleRandFromClass(train_set, labeles_per_class, seed=seed)
    trn_subset_data = subdata[0]
    trn_subset_label = subdata[1].numpy().reshape(-1)

    if cuda_is_avail:
        trn_subset_data = trn_subset_data.cuda()
    z_trn = model.encoder(trn_subset_data)[0]
    z_trn = z_trn.cpu().detach()

    if cuda_is_avail:
        tst_data = tst_data.cuda()
    z_tst = model.encoder(tst_data)[0]
    z_tst = z_tst.cpu().detach()

    svclassifier = SVC(kernel=kernel)  # SVC(kernel='poly', degree=8)
    # trn_subset_label = subdata[1].cpu().numpy().reshape(-1)
    svclassifier.fit(z_trn, trn_subset_label)
    y_pred = svclassifier.predict(z_tst)
    cm = confusion_matrix(tst_label, y_pred)
    acc = accuracy_score(tst_label, y_pred)
    # print(cm)
    # cm = np.random.randint(100, size=(10,10))
    # cm = confusion_matrix(tstlabel, y_pred)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, classes=list(range(10)), normalize=False,
                          title=f'SVM {size}:  kernel: {kernel} , accuracy: {acc}')
    if savefig:
        plt.savefig(f'{log_dir_name}/SVM_results_{size}_{seed if seed != None else ""}.png', dpi=150)
        dump(svclassifier, f'{log_dir_name}/SVM_{size}_{seed if seed != None else ""}.joblib')
        plt.show()
    else:
        plt.show()

# dump(clf, 'filename.joblib')
# clf = load('filename.joblib')