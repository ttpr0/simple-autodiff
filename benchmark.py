import numpy as np
import matplotlib.pyplot as plt
from autodiff.utils import reset_grads, apply_grads
import autodiff as ad
import torch

#*********************************************
# load mnist dataset
#*********************************************

def load_mnist(folder: str):
    with open(folder+"/t10k-images.idx3-ubyte", "rb") as f:
        f.read(16)
        image_size = 28
        num_images = 10000
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        test_img = data.reshape(num_images, 1, image_size, image_size)
    with open(folder+"/train-images.idx3-ubyte", "rb") as f:
        f.read(16)
        image_size = 28
        num_images = 60000
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        train_img = data.reshape(num_images, 1, image_size, image_size)
    with open(folder+"/t10k-labels.idx1-ubyte", "rb") as f:
        f.read(8)
        num_images = 10000
        buf = f.read(num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        test_lbl = data.reshape(num_images,1)
    with open(folder+"/train-labels.idx1-ubyte", "rb") as f:
        f.read(8)
        num_images = 60000
        buf = f.read(num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        train_lbl = data.reshape(num_images,1)
    return train_img, train_lbl, test_img, test_lbl

# load mnist data
train_img0, train_lbl0, test_img0, test_lbl0 = load_mnist("./data")

# normalize images and labels
train_img = train_img0 / 255
test_img = test_img0 / 255
train_lbl = np.zeros((train_lbl0.shape[0], 10))
for i in range(0, train_lbl0.shape[0]):
    train_lbl[i, int(train_lbl0[i])] = 1
test_lbl = np.zeros((test_lbl0.shape[0], 10))
for i in range(0, test_lbl0.shape[0]):
    test_lbl[i, int(test_lbl0[i])] = 1

#*********************************************
# train model (autodiff)
#*********************************************

# initialize trainable weights
weight1 = 2 * np.random.rand(200, 784) - 1
bias1 = 2 * np.random.rand(200, 1) - 1
weight2 = 2 * np.random.rand(100, 200) - 1
bias2 = 2 * np.random.rand(100, 1) - 1
weight3 = 2 * np.random.rand(10,100) - 1

# initialize arrays
weight1 = ad.from_numpy(weight1, track_grads=True)
bias1 = ad.from_numpy(bias1, track_grads=True)
weight2 = ad.from_numpy(weight2, track_grads=True)
bias2 = ad.from_numpy(bias2, track_grads=True)
weight3 = ad.from_numpy(weight3, track_grads=True)

def forward(img: ad.Array) -> ad.Array:
    i1 = ad.reshape(img, (784,1))
    i2 = weight1@i1 + bias1
    i3 = ad.sigmoid(i2)
    i4 = weight2@i3 + bias2
    i5 = ad.sigmoid(i4)
    i6 = weight3@i5
    i7 = ad.softmax(i6)
    return i7

def run_test_set() -> tuple[int, int]:
    right = 0
    wrong = 0
    for i in range(0, 10000):
        img = ad.Array(test_img[i], dtype=np.float32)
        lbl = test_lbl[i]

        out = forward(img)

        out = np.reshape(out.value, (10,))

        index_lbl = np.argmax(lbl)
        index_out = np.argmax(out)

        if index_out == index_lbl:
            right += 1
        else:
            wrong += 1

    return right, wrong

# run training
epochs = 10
pred_results = np.zeros(epochs+1)
right, wrong = run_test_set()
pred_results[0] = (right / (wrong+right)) * 100
for epoch in range(0,epochs):
    in_size = 60000
    batchsize = 30
    permutation = np.random.permutation(in_size)
    loss_acc = 0
    for i in range(0, 60000):
        perm = permutation[i]
        img = ad.Array(train_img[perm], dtype=np.float32)
        lbl = ad.reshape(ad.Array(train_lbl[perm], dtype=np.float32), (10, 1))

        with ad.track_computation():
            output = forward(img)
            loss = ad.mean_squared_error(output, lbl)

        if i%100==0:
            print(f"epoch: {epoch}, iteration: {i}, loss: {loss}")

        loss.backward()

        if i%30==0:
            apply_grads(loss, lr=0.01)
            reset_grads(loss)

    right, wrong = run_test_set()
    pred_results[epoch+1] = (right / (wrong+right)) * 100

# plot results
fig1 = plt.figure()
ax = fig1.add_subplot(111)
x = np.arange(0, epochs+1)
ax.plot(x, pred_results, label="p")
ax.set_title("Predictions")
ax.set_xlabel("epoch")
ax.set_xticks(np.arange(0, epochs+1))
ax.set_ylabel("right predictions [%]")
ax.set_ylim(0, 100)
ax.legend()

# plot tests
fig2 = plt.figure()
for i in range(0, 9):
    img = ad.Array(test_img[i], dtype=np.float32)

    out = forward(img)
    out = np.reshape(out.value, (10,))
    index_out = np.argmax(out)

    ax = fig2.add_subplot(3,3,i+1)

    img = np.reshape(img.value, (28,28))

    ax.imshow(img)
    ax.set_title(f"Prediction: {index_out}")
    ax.set_axis_off()

plt.tight_layout()
plt.show()

#*********************************************
# train model (torch)
#*********************************************

# initialize trainable weights
weight1 = 2 * torch.rand(200, 784) - 1
bias1 = 2 * torch.rand(200) - 1
weight2 = 2 * torch.rand(100, 200) - 1
bias2 = 2 * torch.rand(100) - 1
weight3 = 2 * torch.rand(10,100) - 1

# initialize tensors
weight1 = torch.tensor(weight1, requires_grad=True)
bias1 = torch.tensor(bias1, requires_grad=True)
weight2 = torch.tensor(weight2, requires_grad=True)
bias2 = torch.tensor(bias2, requires_grad=True)
weight3 = torch.tensor(weight3, requires_grad=True)
tensors = [weight1, bias1, weight2, bias2, weight3]

def forward(img):
    i1 = torch.reshape(img, (784,))
    i2 = weight1@i1 + bias1
    i3 = torch.sigmoid(i2)
    i4 = weight2@i3 + bias2
    i5 = torch.sigmoid(i4)
    i6 = weight3@i5
    i7 = torch.softmax(i6, dim=0)
    return i7

def error(output, target):
    return ((target - output)**2).mean()

def run_test_set() -> tuple[int, int]:
    right = 0
    wrong = 0
    for i in range(0, 10000):
        img = torch.tensor(test_img[i], dtype=torch.float32)
        lbl = torch.tensor(test_lbl[i])

        out = forward(img)

        out = torch.reshape(out, (10,))

        index_lbl = torch.argmax(lbl)
        index_out = torch.argmax(out)

        if index_out == index_lbl:
            right += 1
        else:
            wrong += 1

    return right, wrong

# run training
epochs = 10
pred_results = np.zeros(epochs+1)
right, wrong = run_test_set()
pred_results[0] = (right / (wrong+right)) * 100
optim = torch.optim.SGD(tensors, 0.01)
for epoch in range(0,epochs):
    in_size = 60000
    batchsize = 30
    permutation = np.random.permutation(in_size)
    for i in range(0, 60000):
        perm = permutation[i]
        img = torch.tensor(train_img[perm], dtype=torch.float32)
        lbl = torch.tensor(train_lbl[perm], dtype=torch.float32)

        output = forward(img)
        loss = error(output, lbl)

        if i%100==0:
            print(f"epoch: {epoch}, iteration: {i}, loss: {loss}")

        loss.backward()

        if i%30==0:
            optim.step()
            optim.zero_grad()

    right, wrong = run_test_set()
    print(right, wrong)
    pred_results[epoch+1] = (right / (wrong+right)) * 100

# plot results
fig1 = plt.figure()
ax = fig1.add_subplot(111)
x = np.arange(0, epochs+1)
ax.plot(x, pred_results, label="p")
ax.set_title("Predictions")
ax.set_xlabel("epoch")
ax.set_xticks(np.arange(0, epochs+1))
ax.set_ylabel("right predictions [%]")
ax.set_ylim(0, 100)
ax.legend()

# plot tests
fig2 = plt.figure()
for i in range(0, 9):
    img = torch.tensor(test_img[i], dtype=torch.float32)

    out = forward(img)
    out = torch.reshape(out, (10,))
    index_out = torch.argmax(out)

    ax = fig2.add_subplot(3,3,i+1)

    img = np.reshape(img.numpy(), (28,28))

    ax.imshow(img)
    ax.set_title(f"Prediction: {index_out}")
    ax.set_axis_off()

plt.tight_layout()
plt.show()